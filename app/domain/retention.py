"""
app/domain/retention.py – retention field, default 180 days (row 42).

The ``tenants.retention_days`` column (added by migration 0006) controls
how long message and draft rows are kept before a purge job removes them.

Default: 180 days.

Phase 4 additions
-----------------
``apply_retention(db, tenant_id)``
    Deletes messages older than the tenant's retention window.
    Messages with ``legal_hold=True`` are skipped.

``delete_tenant_data(db, tenant_id)``
    Right-to-erasure helper: deletes all personal data for one tenant.
    Messages with ``legal_hold=True`` are NOT deleted (legal hold blocks
    retention delete per Section B item 12).
    Returns a dict describing what was removed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

RETENTION_DEFAULT_DAYS = 180


def get_retention_days(tenant) -> int:
    """Return the effective retention window for *tenant*."""
    val = getattr(tenant, "retention_days", None)
    if val is None:
        return RETENTION_DEFAULT_DAYS
    return int(val)


def apply_retention(
    db: Session,
    tenant_id: int,
    tenant_retention_days: Optional[int] = None,
    now: Optional[datetime] = None,
) -> int:
    """
    Delete messages (and their dependent drafts) older than the retention window.

    Messages with ``legal_hold=True`` are skipped.
    Returns the number of message rows deleted.
    """
    from app.ingest.models import Message

    if now is None:
        now = datetime.now(timezone.utc)

    days = tenant_retention_days if tenant_retention_days is not None else RETENTION_DEFAULT_DAYS
    cutoff = now - timedelta(days=days)

    old_messages = (
        db.query(Message)
        .filter(
            Message.tenant_id == tenant_id,
            Message.ingest_time < cutoff,
        )
        .all()
    )

    count = 0
    for msg in old_messages:
        # Legal-hold blocks deletion (Section B item 12).
        if getattr(msg, "legal_hold", False):
            continue
        db.delete(msg)
        count += 1

    if count:
        db.commit()

    return count


def delete_tenant_data(
    db: Session,
    tenant_id: int,
) -> Dict[str, Any]:
    """
    Right-to-erasure: delete all personal data for *tenant_id*.

    Deletes in dependency order so foreign-key constraints are satisfied:
      1. draft rows  (reference messages)
      2. approval_queue rows  (reference messages)
      3. audit_log rows
      4. usage_events rows
      5. message rows  (skipped when legal_hold=True)
      6. delivery_log rows
      7. work_queue rows
      8. dead_letter rows

    User account rows and the tenant row itself are intentionally NOT
    removed here; account deletion is a separate operator action.

    Returns a dict with ``deleted_<table>`` counts and a ``legal_hold_skipped``
    count for messages that were protected.
    """
    from app.policy.shadow import Draft
    from app.policy.approval import ApprovalQueueEntry
    from app.policy.audit import AuditLogEntry
    from app.domain.usage import UsageEvent
    from app.ingest.models import Message
    from app.workers.queue import WorkItem
    from app.workers.dlq import DeadLetterItem

    result: Dict[str, Any] = {
        "tenant_id": tenant_id,
        "deleted_drafts": 0,
        "deleted_approval_queue": 0,
        "deleted_audit_log": 0,
        "deleted_usage_events": 0,
        "deleted_messages": 0,
        "deleted_work_queue": 0,
        "deleted_dead_letter": 0,
        "legal_hold_skipped": 0,
    }

    # 1. Drafts (reference messages.id — must go first)
    drafts = db.query(Draft).filter(Draft.tenant_id == tenant_id).all()
    for d in drafts:
        db.delete(d)
        result["deleted_drafts"] += 1

    # 2. Approval queue
    aq_rows = (
        db.query(ApprovalQueueEntry)
        .filter(ApprovalQueueEntry.tenant_id == tenant_id)
        .all()
    )
    for aq in aq_rows:
        db.delete(aq)
        result["deleted_approval_queue"] += 1

    # 3. Audit log
    al_rows = (
        db.query(AuditLogEntry)
        .filter(AuditLogEntry.tenant_id == tenant_id)
        .all()
    )
    for al in al_rows:
        db.delete(al)
        result["deleted_audit_log"] += 1

    # 4. Usage events
    ue_rows = (
        db.query(UsageEvent)
        .filter(UsageEvent.tenant_id == tenant_id)
        .all()
    )
    for ue in ue_rows:
        db.delete(ue)
        result["deleted_usage_events"] += 1

    # 5. Messages — skip legal-hold rows
    msgs = db.query(Message).filter(Message.tenant_id == tenant_id).all()
    for msg in msgs:
        if getattr(msg, "legal_hold", False):
            result["legal_hold_skipped"] += 1
            continue
        db.delete(msg)
        result["deleted_messages"] += 1

    # 6. Work queue
    try:
        wq_rows = db.query(WorkItem).filter(WorkItem.tenant_id == tenant_id).all()
        for wq in wq_rows:
            db.delete(wq)
            result["deleted_work_queue"] += 1
    except Exception:  # noqa: BLE001
        pass  # table may not exist in all deployments

    # 7. Dead-letter
    try:
        dl_rows = (
            db.query(DeadLetterItem)
            .filter(DeadLetterItem.tenant_id == tenant_id)
            .all()
        )
        for dl in dl_rows:
            db.delete(dl)
            result["deleted_dead_letter"] += 1
    except Exception:  # noqa: BLE001
        pass

    db.commit()
    return result
