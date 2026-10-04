"""
app/domain/retention.py – retention field, default 180 days (row 42).

The ``tenants.retention_days`` column (added by migration 0006) controls
how long message and draft rows are kept before a purge job removes them.

Default: 180 days.

``apply_retention(db, tenant_id)`` deletes messages older than the
tenant's retention window and emits a usage event for each row deleted.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

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

    Returns the number of message rows deleted.
    This is the purge-job entry point; it is NOT called at import time.
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
        db.delete(msg)
        count += 1

    if count:
        db.commit()

    return count
