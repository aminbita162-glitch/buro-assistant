"""
app/web/desk.py – Operator desk router (Phase 3 / follow-up).

Rows closed:
  23 – dashboard counts: received, classified, drafted, sent, held, failed
  24 – department queues (stored field, not subject keyword only)

All endpoints require a valid session token (Bearer).
All queries are scoped to the authenticated user's tenant_id.

Mounted at /desk by app/main.py via app.include_router().

Endpoints
---------
GET  /desk/dashboard              Row 23 – message and draft counts by state
GET  /desk/queue                  Inbound messages (all states), newest first
GET  /desk/queue/{department}     Row 24 – messages filtered by stored department field
GET  /desk/decisions              Decisions for the tenant, newest first
GET  /desk/drafts                 Drafts for the tenant, newest first
GET  /desk/approval               Pending approval queue entries
POST /desk/approval/{id}/approve  Approve a pending entry (calls resolve())
POST /desk/approval/{id}/reject   Reject a pending entry  (calls resolve())
GET  /desk/audit                  Recent audit log entries (default limit 100)
GET  /desk/tasks                  Active task list (mirrors /tasks, tenant-scoped)
GET  /desk/quota                  Today's token quota and usage for the tenant
GET  /desk/cost                   Cumulative cost summary from usage_events

All imports from app.main are deferred to function bodies to avoid the
circular-import that would arise from app.main importing this module at
module load time.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query

router = APIRouter(prefix="/desk", tags=["desk"])


# ---------------------------------------------------------------------------
# Shared auth helper
# ---------------------------------------------------------------------------

def _open_db_and_auth(authorization: Optional[str]):
    """
    Return ``(db, user)``.  Caller is responsible for closing ``db`` in a
    ``finally`` block.  Raises ``HTTPException(401)`` when not authenticated.
    """
    # Deferred to avoid circular import.
    from app.main import SessionLocal, require_current_user
    db = SessionLocal()
    try:
        user = require_current_user(db, authorization)
    except HTTPException:
        db.close()
        raise
    return db, user


# ---------------------------------------------------------------------------
# Row 23 – Dashboard counts
# ---------------------------------------------------------------------------

@router.get("/dashboard")
def dashboard(authorization: Optional[str] = Header(default=None)):
    """
    Return message and draft state counts for the authenticated tenant (row 23).

    Message states: received (new), classified, quarantine, duplicate, failed.
    Draft states:   draft, shadow, sent, approved, rejected.
    Held:           pending approval-queue entries.
    """
    from app.main import safe_db_error_message
    from app.ingest.models import Message
    from app.policy.shadow import Draft
    from app.policy.approval import ApprovalQueueEntry

    db, user = _open_db_and_auth(authorization)
    try:
        tid = user.tenant_id

        def _msg(state: str) -> int:
            return (
                db.query(Message)
                .filter(Message.tenant_id == tid, Message.state == state)
                .count()
            )

        def _draft(state: str) -> int:
            return (
                db.query(Draft)
                .filter(Draft.tenant_id == tid, Draft.state == state)
                .count()
            )

        held = (
            db.query(ApprovalQueueEntry)
            .filter(
                ApprovalQueueEntry.tenant_id == tid,
                ApprovalQueueEntry.state == "pending",
            )
            .count()
        )

        return {
            "tenant_id": tid,
            "messages": {
                "received":   _msg("new"),
                "classified": _msg("classified"),
                "quarantine": _msg("quarantine"),
                "duplicate":  _msg("duplicate"),
                "failed":     _msg("failed"),
            },
            "drafts": {
                "drafted":  _draft("draft") + _draft("shadow"),
                "sent":     _draft("sent"),
                "approved": _draft("approved"),
                "rejected": _draft("rejected"),
            },
            "held": held,
        }
    except HTTPException:
        raise
    except Exception as exc:
        from app.main import safe_db_error_message
        raise HTTPException(status_code=500, detail=safe_db_error_message(exc))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Inbound message queue
# ---------------------------------------------------------------------------

@router.get("/queue")
def inbound_queue(
    state: Optional[str] = Query(default=None, description="Filter by message state"),
    limit: int = Query(default=50, ge=1, le=500),
    authorization: Optional[str] = Header(default=None),
):
    """Return inbound messages for the tenant, newest first."""
    from app.main import safe_db_error_message
    from app.ingest.models import Message

    db, user = _open_db_and_auth(authorization)
    try:
        q = db.query(Message).filter(Message.tenant_id == user.tenant_id)
        if state:
            q = q.filter(Message.state == state)
        rows = q.order_by(Message.id.desc()).limit(limit).all()
        return {"count": len(rows), "messages": [_ser_message(m) for m in rows]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_db_error_message(exc))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Row 24 – Department queue  (stored field, with subject-keyword fallback)
# ---------------------------------------------------------------------------

@router.get("/queue/{department}")
def department_queue(
    department: str,
    limit: int = Query(default=50, ge=1, le=500),
    authorization: Optional[str] = Header(default=None),
):
    """
    Return messages for the *department* (row 24).

    Matches the stored ``messages.department`` column first.  When a message
    has no stored department, falls back to a subject-keyword contains check
    so that messages ingested before Phase 3 are still surfaced.
    """
    from app.main import safe_db_error_message
    from app.ingest.models import Message
    from sqlalchemy import func, or_

    db, user = _open_db_and_auth(authorization)
    try:
        dept_kw = department.strip().lower()
        rows = (
            db.query(Message)
            .filter(
                Message.tenant_id == user.tenant_id,
                or_(
                    func.lower(Message.department) == dept_kw,
                    func.lower(Message.subject_normalized).contains(dept_kw),
                ),
            )
            .order_by(Message.id.desc())
            .limit(limit)
            .all()
        )
        return {
            "department": department,
            "count": len(rows),
            "messages": [_ser_message(m) for m in rows],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_db_error_message(exc))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

@router.get("/decisions")
def decisions(
    limit: int = Query(default=50, ge=1, le=500),
    authorization: Optional[str] = Header(default=None),
):
    """Return triage/supervisor decisions for the tenant, newest first."""
    from app.main import safe_db_error_message, engine
    from sqlalchemy import inspect as sa_inspect, text as sa_text

    db, user = _open_db_and_auth(authorization)
    try:
        if "decisions" not in sa_inspect(engine).get_table_names():
            return {"count": 0, "decisions": []}

        rows = db.execute(
            sa_text(
                "SELECT id, agent, action, department, language, urgency, "
                "confidence, rule_hit, reason, decision_hash, created_at "
                "FROM decisions WHERE tenant_id = :tid "
                "ORDER BY id DESC LIMIT :lim"
            ),
            {"tid": user.tenant_id, "lim": limit},
        ).fetchall()

        return {
            "count": len(rows),
            "decisions": [
                {
                    "id": r[0], "agent": r[1], "action": r[2],
                    "department": r[3], "language": r[4], "urgency": r[5],
                    "confidence": r[6], "rule_hit": r[7], "reason": r[8],
                    "decision_hash": r[9], "created_at": str(r[10]),
                }
                for r in rows
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_db_error_message(exc))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Drafts
# ---------------------------------------------------------------------------

@router.get("/drafts")
def drafts(
    state: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    authorization: Optional[str] = Header(default=None),
):
    """Return drafts for the tenant, newest first."""
    from app.main import safe_db_error_message
    from app.policy.shadow import Draft

    db, user = _open_db_and_auth(authorization)
    try:
        q = db.query(Draft).filter(Draft.tenant_id == user.tenant_id)
        if state:
            q = q.filter(Draft.state == state)
        rows = q.order_by(Draft.id.desc()).limit(limit).all()
        return {"count": len(rows), "drafts": [_ser_draft(d) for d in rows]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_db_error_message(exc))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Approval queue view + approve / reject actions
# ---------------------------------------------------------------------------

@router.get("/approval")
def approval_queue(
    limit: int = Query(default=50, ge=1, le=500),
    authorization: Optional[str] = Header(default=None),
):
    """Return pending approval queue entries for the tenant."""
    from app.main import safe_db_error_message
    from app.policy.approval import ApprovalQueueEntry

    db, user = _open_db_and_auth(authorization)
    try:
        rows = (
            db.query(ApprovalQueueEntry)
            .filter(
                ApprovalQueueEntry.tenant_id == user.tenant_id,
                ApprovalQueueEntry.state == "pending",
            )
            .order_by(ApprovalQueueEntry.created_at.desc())
            .limit(limit)
            .all()
        )
        return {"count": len(rows), "entries": [_ser_approval(e) for e in rows]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_db_error_message(exc))
    finally:
        db.close()


@router.post("/approval/{entry_id}/approve")
def approve_entry(
    entry_id: int,
    authorization: Optional[str] = Header(default=None),
):
    """
    Approve a pending approval queue entry.

    Calls ``policy.approval.resolve()`` which writes the decision back and
    enqueues the approved draft for delivery (if tenant policy allows send).
    Returns the updated entry or 404 when not found / wrong tenant.
    """
    from app.main import safe_db_error_message
    from app.policy.approval import resolve

    db, user = _open_db_and_auth(authorization)
    try:
        entry = resolve(db, entry_id, user.tenant_id, "approved",
                        resolved_by=user.email or str(user.id))
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail="Entry not found or already resolved",
            )
        return {"status": "approved", "entry": _ser_approval(entry)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_db_error_message(exc))
    finally:
        db.close()


@router.post("/approval/{entry_id}/reject")
def reject_entry(
    entry_id: int,
    authorization: Optional[str] = Header(default=None),
):
    """
    Reject a pending approval queue entry.

    Calls ``policy.approval.resolve()`` with decision="rejected".
    Returns the updated entry or 404 when not found / wrong tenant.
    """
    from app.main import safe_db_error_message
    from app.policy.approval import resolve

    db, user = _open_db_and_auth(authorization)
    try:
        entry = resolve(db, entry_id, user.tenant_id, "rejected",
                        resolved_by=user.email or str(user.id))
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail="Entry not found or already resolved",
            )
        return {"status": "rejected", "entry": _ser_approval(entry)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_db_error_message(exc))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Audit log view
# ---------------------------------------------------------------------------

@router.get("/audit")
def audit_log(
    limit: int = Query(default=100, ge=1, le=1000),
    authorization: Optional[str] = Header(default=None),
):
    """Return recent audit log entries for the tenant."""
    from app.main import safe_db_error_message
    from app.policy.audit import recent_for_tenant

    db, user = _open_db_and_auth(authorization)
    try:
        rows = recent_for_tenant(db, user.tenant_id, limit=limit)
        return {
            "count": len(rows),
            "entries": [
                {
                    "id": e.id,
                    "event": e.event,
                    "actor": e.actor,
                    "message_id": e.message_id,
                    "detail": e.detail,
                    "created_at": e.created_at.isoformat(),
                }
                for e in rows
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_db_error_message(exc))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Quota and cost summary
# ---------------------------------------------------------------------------

@router.get("/quota")
def desk_quota(authorization: Optional[str] = Header(default=None)):
    """
    Return today's token quota row for the authenticated tenant.

    Fields: quota_date, tokens_used, cost_usd_used, daily_token_quota,
            tokens_remaining.
    """
    from app.main import safe_db_error_message
    from app.workers.quota import get_or_create_quota, DEFAULT_DAILY_TOKEN_QUOTA

    db, user = _open_db_and_auth(authorization)
    try:
        row = get_or_create_quota(db, user.tenant_id)
        limit = DEFAULT_DAILY_TOKEN_QUOTA
        return {
            "tenant_id": user.tenant_id,
            "quota_date": str(row.quota_date),
            "daily_token_quota": limit,
            "tokens_used": row.tokens_used,
            "tokens_remaining": max(0, limit - row.tokens_used),
            "cost_usd_used": row.cost_usd_used,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_db_error_message(exc))
    finally:
        db.close()


@router.get("/cost")
def desk_cost(
    limit: int = Query(default=100, ge=1, le=1000),
    authorization: Optional[str] = Header(default=None),
):
    """
    Return a cost summary from usage_events for the authenticated tenant.

    Aggregates total tokens and total cost_usd across all usage_events rows,
    and returns the most recent *limit* individual events.
    """
    from app.main import safe_db_error_message
    from app.domain.usage import events_for_tenant

    db, user = _open_db_and_auth(authorization)
    try:
        events = events_for_tenant(db, user.tenant_id, limit=limit)
        total_tokens = sum(
            e.quantity for e in events if (e.unit or "") == "tokens"
        )
        total_cost = sum(
            e.cost_usd for e in events if e.cost_usd is not None
        )
        return {
            "tenant_id": user.tenant_id,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "event_count": len(events),
            "events": [
                {
                    "id": e.id,
                    "event_type": e.event_type,
                    "quantity": e.quantity,
                    "unit": e.unit,
                    "cost_usd": e.cost_usd,
                    "actor": e.actor,
                    "created_at": e.created_at.isoformat(),
                }
                for e in events
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_db_error_message(exc))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Task list (existing task list, re-exposed at desk path)
# ---------------------------------------------------------------------------

@router.get("/tasks")
def desk_tasks(authorization: Optional[str] = Header(default=None)):
    """Return the active task list for the authenticated user (tenant-scoped)."""
    from app.main import safe_db_error_message, get_active_tasks, serialize_task

    db, user = _open_db_and_auth(authorization)
    try:
        tasks = get_active_tasks(db, user.id, user.tenant_id)
        return {"count": len(tasks), "tasks": [serialize_task(t) for t in tasks]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=safe_db_error_message(exc))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Serialisers
# ---------------------------------------------------------------------------

def _ser_message(m) -> dict:
    return {
        "id": m.id,
        "provider": m.provider,
        "provider_message_id": m.provider_message_id,
        "subject_normalized": m.subject_normalized,
        "message_id_header": m.message_id_header,
        "department": getattr(m, "department", None),
        "state": m.state,
        "attachment_state": m.attachment_state,
        "ingest_time": m.ingest_time.isoformat() if m.ingest_time else None,
        "tenant_id": m.tenant_id,
    }


def _ser_draft(d) -> dict:
    return {
        "id": d.id,
        "subject": d.subject,
        "body": d.body,
        "template_id": d.template_id,
        "language": d.language,
        "decision_hash": d.decision_hash,
        "state": d.state,
        "created_at": d.created_at.isoformat() if d.created_at else None,
        "message_id": d.message_id,
        "tenant_id": d.tenant_id,
    }


def _ser_approval(e) -> dict:
    return {
        "id": e.id,
        "subject": e.subject,
        "body": e.body,
        "template_id": e.template_id,
        "state": e.state,
        "reason": e.reason,
        "created_at": e.created_at.isoformat() if e.created_at else None,
        "message_id": e.message_id,
        "tenant_id": e.tenant_id,
    }
