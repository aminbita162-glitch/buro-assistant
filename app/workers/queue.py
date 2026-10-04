"""
app/workers/queue.py – work queue with priority lanes and backpressure (rows 27, 29).

The queue is a DB-backed list of work items stored in the ``work_queue`` table
(created by migration 0005).  Items are dequeued by the worker in priority
order: critical → high → medium → low.

Backpressure (row 27)
---------------------
When the total pending queue depth for a tenant reaches ``queue_depth_cap``
(default 500, from tenant policy config), new items are rejected with
:class:`BackpressureError` instead of being enqueued.

Priority lanes (row 29)
-----------------------
Lane ordering: critical=0, high=1, medium=2, low=3.
``dequeue_next`` always returns the highest-priority pending item for a tenant.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Session

from app.main import Base
from app.workers.quota import DEFAULT_QUEUE_DEPTH_CAP


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class WorkItem(Base):
    __tablename__ = "work_queue"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=True, index=True)

    # Priority lane: critical | high | medium | low
    priority = Column(String, nullable=False, default="medium", index=True)

    # Numeric lane for ORDER BY (lower = higher priority)
    lane = Column(Integer, nullable=False, default=2, index=True)

    # State: pending | processing | done | failed | dead
    state = Column(String, nullable=False, default="pending", index=True)

    payload = Column(Text, nullable=True)    # JSON string, opaque to the queue
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Priority → lane mapping (row 29)
# ---------------------------------------------------------------------------

PRIORITY_LANE: Dict[str, int] = {
    "critical": 0,
    "high":     1,
    "medium":   2,
    "low":      3,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BackpressureError(Exception):
    """Raised when the tenant queue depth cap is exceeded (row 27)."""


# ---------------------------------------------------------------------------
# Queue helpers
# ---------------------------------------------------------------------------

def _depth(db: Session, tenant_id: int) -> int:
    return (
        db.query(WorkItem)
        .filter(WorkItem.tenant_id == tenant_id, WorkItem.state == "pending")
        .count()
    )


def enqueue_work(
    db: Session,
    tenant_id: int,
    priority: str = "medium",
    message_id: Optional[int] = None,
    payload: Optional[str] = None,
    policy_config: Optional[Dict[str, Any]] = None,
) -> WorkItem:
    """
    Add a work item to the queue.

    Raises :class:`BackpressureError` when the tenant's pending depth
    reaches ``queue_depth_cap`` (row 27).
    Inserts into the correct priority lane (row 29).
    """
    cap = int((policy_config or {}).get("queue_depth_cap", DEFAULT_QUEUE_DEPTH_CAP))
    depth = _depth(db, tenant_id)
    if depth >= cap:
        raise BackpressureError(
            f"Tenant {tenant_id} queue depth {depth} has reached cap {cap}"
        )

    lane = PRIORITY_LANE.get(priority, 2)
    item = WorkItem(
        tenant_id=tenant_id,
        message_id=message_id,
        priority=priority,
        lane=lane,
        state="pending",
        payload=payload,
        created_at=datetime.now(timezone.utc),
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def dequeue_next(db: Session, tenant_id: int) -> Optional[WorkItem]:
    """
    Return and mark-as-processing the next highest-priority pending item.
    Returns None when the queue is empty.
    Priority lane order: critical (0) < high (1) < medium (2) < low (3).
    """
    item = (
        db.query(WorkItem)
        .filter(WorkItem.tenant_id == tenant_id, WorkItem.state == "pending")
        .order_by(WorkItem.lane.asc(), WorkItem.id.asc())
        .first()
    )
    if item is None:
        return None
    item.state = "processing"
    item.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(item)
    return item


def complete_item(db: Session, item_id: int) -> Optional[WorkItem]:
    """Mark a work item as done."""
    item = db.query(WorkItem).filter(WorkItem.id == item_id).first()
    if item:
        item.state = "done"
        item.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(item)
    return item


def fail_item(db: Session, item_id: int, reason: str = "") -> Optional[WorkItem]:
    """Mark a work item as failed (candidate for DLQ)."""
    item = db.query(WorkItem).filter(WorkItem.id == item_id).first()
    if item:
        item.state = "failed"
        item.payload = reason or item.payload
        item.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(item)
    return item


def queue_depth(db: Session, tenant_id: int) -> int:
    """Return the number of pending items for a tenant."""
    return _depth(db, tenant_id)
