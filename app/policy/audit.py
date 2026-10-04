"""
app/policy/audit.py – append-only audit log (row 22).

The audit log records every significant policy event.  Rows are never
updated or deleted; only INSERT is performed.  A soft-delete flag is
deliberately absent to keep the log append-only.

Event categories
----------------
message_ingested, triage_decided, draft_created, draft_approved,
draft_rejected, message_sent, message_held, message_failed,
shadow_draft_stored, approval_requested
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Session

from app.main import Base


# ---------------------------------------------------------------------------
# ORM model (append-only – no update/delete operations)
# ---------------------------------------------------------------------------

class AuditLogEntry(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)

    # Event classification
    event = Column(String, nullable=False, index=True)

    # Optional references
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=True, index=True)
    actor = Column(String, nullable=True)       # user email or "system"

    # Free-form detail (JSON string or plain text)
    detail = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, index=True)


# ---------------------------------------------------------------------------
# Append helper — the ONLY write path for the audit log
# ---------------------------------------------------------------------------

def log_event(
    db: Session,
    tenant_id: int,
    event: str,
    actor: str = "system",
    message_id: Optional[int] = None,
    detail: Optional[str] = None,
) -> AuditLogEntry:
    """
    Write one immutable audit log row.

    This is the sole write path.  There is no update or delete function.
    """
    entry = AuditLogEntry(
        tenant_id=tenant_id,
        event=event,
        actor=actor,
        message_id=message_id,
        detail=detail,
        created_at=datetime.now(timezone.utc),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def recent_for_tenant(
    db: Session,
    tenant_id: int,
    limit: int = 100,
) -> list:
    """Return the most recent audit log entries for a tenant."""
    return (
        db.query(AuditLogEntry)
        .filter(AuditLogEntry.tenant_id == tenant_id)
        .order_by(AuditLogEntry.created_at.desc())
        .limit(limit)
        .all()
    )
