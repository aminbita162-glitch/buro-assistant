"""
app/policy/approval.py – human approval queue (row 21).

The approval queue stores one row per draft that requires human sign-off
before it may be sent.  A draft enters the queue when:
  - auto_reply is disabled (default), or
  - Leila routes to request_human.

Queue entries are fulfilled (approved / rejected) by operators.
The ORM model lives here; the table is created by migration 0004.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Session

from app.main import Base


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class ApprovalQueueEntry(Base):
    __tablename__ = "approval_queue"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)

    # The message that generated this draft.
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=True, index=True)

    # The draft content (denormalised for display without joining drafts table).
    subject = Column(String, nullable=False, default="")
    body = Column(String, nullable=False, default="")
    template_id = Column(String, nullable=True)

    # Queue state: pending | approved | rejected
    state = Column(String, nullable=False, default="pending")

    # Reason this entry is in the queue.
    reason = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by = Column(String, nullable=True)   # operator identifier / user email


# ---------------------------------------------------------------------------
# Queue helpers
# ---------------------------------------------------------------------------

def enqueue(
    db: Session,
    tenant_id: int,
    subject: str,
    body: str,
    template_id: Optional[str] = None,
    message_id: Optional[int] = None,
    reason: str = "pending_human_review",
) -> ApprovalQueueEntry:
    """Create and persist a new approval queue entry."""
    entry = ApprovalQueueEntry(
        tenant_id=tenant_id,
        message_id=message_id,
        subject=subject,
        body=body,
        template_id=template_id,
        state="pending",
        reason=reason,
        created_at=datetime.now(timezone.utc),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def resolve(
    db: Session,
    entry_id: int,
    tenant_id: int,
    decision: str,              # "approved" | "rejected"
    resolved_by: str = "",
) -> Optional[ApprovalQueueEntry]:
    """
    Resolve an approval queue entry.

    Returns the updated entry, or None if not found / wrong tenant.
    """
    if decision not in ("approved", "rejected"):
        raise ValueError(f"Invalid decision: {decision!r}")

    entry = (
        db.query(ApprovalQueueEntry)
        .filter(
            ApprovalQueueEntry.id == entry_id,
            ApprovalQueueEntry.tenant_id == tenant_id,
            ApprovalQueueEntry.state == "pending",
        )
        .first()
    )
    if not entry:
        return None

    entry.state = decision
    entry.resolved_at = datetime.now(timezone.utc)
    entry.resolved_by = resolved_by
    db.commit()
    db.refresh(entry)
    return entry


def pending_for_tenant(
    db: Session,
    tenant_id: int,
) -> list:
    """Return all pending entries for a tenant, newest first."""
    return (
        db.query(ApprovalQueueEntry)
        .filter(
            ApprovalQueueEntry.tenant_id == tenant_id,
            ApprovalQueueEntry.state == "pending",
        )
        .order_by(ApprovalQueueEntry.created_at.desc())
        .all()
    )
