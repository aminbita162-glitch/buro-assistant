"""
app/policy/audit.py – append-only audit log with hash chain (Phase 5).

The audit log records every significant policy event.  Rows are never
updated or deleted; only INSERT is performed.  A soft-delete flag is
deliberately absent to keep the log append-only.

Hash chain
----------
Each row stores the SHA-256 hex digest of the previous row's canonical
representation (id + tenant_id + event + actor + detail + created_at ISO
string).  The very first row for any sequence stores the sentinel value
"0000000000000000000000000000000000000000000000000000000000000000"
(64 zeros) as its prev_hash.

A changed row breaks the chain: verify_chain() recomputes every hash
in insertion order and returns the ids of rows where the stored prev_hash
does not match the recomputed value.

Event categories
----------------
message_ingested, triage_decided, draft_created, draft_approved,
draft_rejected, message_sent, message_held, message_failed,
shadow_draft_stored, approval_requested
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Session

from app.main import Base


# ---------------------------------------------------------------------------
# Sentinel value for the first row in a chain
# ---------------------------------------------------------------------------

ZERO_HASH = "0" * 64


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

    # Hash chain: SHA-256 of the previous row's canonical fields.
    # Rows written before Phase 5 will have NULL here; they are treated as
    # pre-chain and are excluded from chain verification.
    prev_hash = Column(String(64), nullable=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _canonical(entry: AuditLogEntry) -> str:
    """Return a stable string representation of an entry for hashing."""
    ts = entry.created_at.isoformat() if entry.created_at else ""
    return f"{entry.id}|{entry.tenant_id}|{entry.event}|{entry.actor}|{entry.detail}|{ts}"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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

    The prev_hash is computed from the most recent row for the same tenant
    (ordered by id).  If no previous row exists, ZERO_HASH is used.
    """
    # Determine the hash of the previous row for this tenant.
    prev_entry = (
        db.query(AuditLogEntry)
        .filter(AuditLogEntry.tenant_id == tenant_id)
        .filter(AuditLogEntry.prev_hash.isnot(None))
        .order_by(AuditLogEntry.id.desc())
        .first()
    )
    if prev_entry is None:
        prev_hash = ZERO_HASH
    else:
        prev_hash = _sha256(_canonical(prev_entry))

    entry = AuditLogEntry(
        tenant_id=tenant_id,
        event=event,
        actor=actor,
        message_id=message_id,
        detail=detail,
        created_at=datetime.now(timezone.utc),
        prev_hash=prev_hash,
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


# ---------------------------------------------------------------------------
# Chain verification
# ---------------------------------------------------------------------------

def verify_chain(db: Session, tenant_id: int) -> List[int]:
    """
    Verify the hash chain for a tenant.

    Returns a list of entry ids where the stored prev_hash does not match
    the expected value.  An empty list means the chain is intact.

    Rows with prev_hash IS NULL are pre-chain rows (written before Phase 5)
    and are skipped.
    """
    rows = (
        db.query(AuditLogEntry)
        .filter(AuditLogEntry.tenant_id == tenant_id)
        .filter(AuditLogEntry.prev_hash.isnot(None))
        .order_by(AuditLogEntry.id.asc())
        .all()
    )

    broken: List[int] = []
    expected_prev: Optional[str] = None  # hash of the row just before current

    for i, row in enumerate(rows):
        if i == 0:
            # First chained row must carry ZERO_HASH as its prev_hash.
            if row.prev_hash != ZERO_HASH:
                broken.append(row.id)
        else:
            prev_row = rows[i - 1]
            expected = _sha256(_canonical(prev_row))
            if row.prev_hash != expected:
                broken.append(row.id)

        # Track the expected hash of this row for the next iteration
        # (used implicitly through rows[i-1] above).

    return broken
