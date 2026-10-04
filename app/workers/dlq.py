"""
app/workers/dlq.py – dead-letter queue and replay (row 28).

Failed work items are moved to the ``dead_letter`` table after exhausting
retry attempts.  Operators can replay a dead-letter item by calling
:func:`replay`, which re-enqueues it with a reset retry count.

Dead-letter table columns
-------------------------
id, tenant_id, message_id (FK), original_item_id, priority, payload,
failure_reason, failed_at, replayed_at (nullable), replay_count
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Session

from app.main import Base


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class DeadLetterItem(Base):
    __tablename__ = "dead_letter"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=True, index=True)

    # Reference to the original work_queue row (may be gone after cleanup).
    original_item_id = Column(Integer, nullable=True)
    priority = Column(String, nullable=False, default="medium")
    payload = Column(Text, nullable=True)

    failure_reason = Column(Text, nullable=True)
    failed_at = Column(DateTime(timezone=True), nullable=False)

    replayed_at = Column(DateTime(timezone=True), nullable=True)
    replay_count = Column(Integer, nullable=False, default=0)


# ---------------------------------------------------------------------------
# DLQ helpers (row 28)
# ---------------------------------------------------------------------------

def send_to_dlq(
    db: Session,
    tenant_id: int,
    priority: str = "medium",
    payload: Optional[str] = None,
    failure_reason: str = "",
    message_id: Optional[int] = None,
    original_item_id: Optional[int] = None,
) -> DeadLetterItem:
    """
    Move a failed item to the dead-letter queue.

    Called by the worker after a work item has failed and exhausted retries.
    """
    entry = DeadLetterItem(
        tenant_id=tenant_id,
        message_id=message_id,
        original_item_id=original_item_id,
        priority=priority,
        payload=payload,
        failure_reason=failure_reason,
        failed_at=datetime.now(timezone.utc),
        replay_count=0,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def replay(
    db: Session,
    dlq_item_id: int,
    tenant_id: int,
    policy_config: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """
    Replay a dead-letter item by re-enqueuing it into the work queue.

    Returns the new :class:`~app.workers.queue.WorkItem`, or ``None`` if
    the DLQ entry was not found / belongs to a different tenant.
    Row 28: dead-letter queue and replay.
    """
    from app.workers.queue import enqueue_work

    entry = (
        db.query(DeadLetterItem)
        .filter(
            DeadLetterItem.id == dlq_item_id,
            DeadLetterItem.tenant_id == tenant_id,
        )
        .first()
    )
    if entry is None:
        return None

    # Re-enqueue with original priority.
    work_item = enqueue_work(
        db,
        tenant_id=entry.tenant_id,
        priority=entry.priority,
        message_id=entry.message_id,
        payload=entry.payload,
        policy_config=policy_config,
    )

    # Mark replayed.
    entry.replayed_at = datetime.now(timezone.utc)
    entry.replay_count += 1
    db.commit()
    db.refresh(entry)

    return work_item


def pending_dlq(db: Session, tenant_id: int) -> List[DeadLetterItem]:
    """Return all dead-letter items not yet replayed, newest first."""
    return (
        db.query(DeadLetterItem)
        .filter(
            DeadLetterItem.tenant_id == tenant_id,
            DeadLetterItem.replayed_at.is_(None),
        )
        .order_by(DeadLetterItem.failed_at.desc())
        .all()
    )
