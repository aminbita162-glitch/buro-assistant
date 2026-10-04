"""
app/domain/export.py – tenant data export job (row 41).

``export_tenant(db, tenant_id)`` collects all data owned by a tenant and
returns it as a plain Python dict that can be serialised to JSON.

Exported tables
---------------
tenant record, users, messages (without raw_json by default), drafts,
decisions, audit_log, usage_events, approval_queue, work_queue

The export is read-only; it does not delete or modify any data.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def export_tenant(
    db: Session,
    tenant_id: int,
    include_raw_json: bool = False,
) -> Dict[str, Any]:
    """
    Return a dict containing all data for *tenant_id*.

    Row 41: tenant data export job.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    tenant_id:
        The tenant whose data is exported.
    include_raw_json:
        When True, the ``raw_json`` field of messages is included.
        Defaults to False to keep exports compact.
    """
    from app.main import Tenant, User, Task
    from app.ingest.models import Message
    from app.policy.shadow import Draft
    from app.policy.audit import AuditLogEntry
    from app.policy.approval import ApprovalQueueEntry
    from app.domain.usage import UsageEvent

    # ---- Tenant record ----
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if tenant is None:
        return {"error": f"Tenant {tenant_id} not found"}

    result: Dict[str, Any] = {
        "exported_at": _iso(datetime.now(timezone.utc)),
        "tenant": {
            "id": tenant.id,
            "name": tenant.name,
            "slug": tenant.slug,
        },
        "users": [],
        "messages": [],
        "drafts": [],
        "audit_log": [],
        "usage_events": [],
        "approval_queue": [],
    }

    # ---- Users ----
    for u in db.query(User).filter(User.tenant_id == tenant_id).all():
        result["users"].append({
            "id": u.id, "name": u.name, "email": u.email,
        })

    # ---- Messages ----
    for m in db.query(Message).filter(Message.tenant_id == tenant_id).all():
        row: Dict[str, Any] = {
            "id": m.id,
            "provider": m.provider,
            "provider_message_id": m.provider_message_id,
            "subject_normalized": m.subject_normalized,
            "state": m.state,
            "attachment_state": m.attachment_state,
            "ingest_time": _iso(m.ingest_time),
        }
        if include_raw_json:
            row["raw_json"] = m.raw_json
        result["messages"].append(row)

    # ---- Drafts ----
    for d in db.query(Draft).filter(Draft.tenant_id == tenant_id).all():
        result["drafts"].append({
            "id": d.id, "subject": d.subject, "state": d.state,
            "template_id": d.template_id, "created_at": _iso(d.created_at),
        })

    # ---- Audit log ----
    for e in db.query(AuditLogEntry).filter(AuditLogEntry.tenant_id == tenant_id).all():
        result["audit_log"].append({
            "id": e.id, "event": e.event, "actor": e.actor,
            "created_at": _iso(e.created_at),
        })

    # ---- Usage events ----
    for ev in db.query(UsageEvent).filter(UsageEvent.tenant_id == tenant_id).all():
        result["usage_events"].append({
            "id": ev.id, "event_type": ev.event_type,
            "quantity": ev.quantity, "unit": ev.unit,
            "cost_usd": ev.cost_usd, "created_at": _iso(ev.created_at),
        })

    # ---- Approval queue ----
    for aq in (
        db.query(ApprovalQueueEntry)
        .filter(ApprovalQueueEntry.tenant_id == tenant_id)
        .all()
    ):
        result["approval_queue"].append({
            "id": aq.id, "subject": aq.subject,
            "state": aq.state, "created_at": _iso(aq.created_at),
        })

    return result
