"""
app/domain/apikeys.py – scoped API keys, stored hashed (row 44).

API keys are issued per-tenant with an optional scope string.  Only the
SHA-256 hash of the raw key is stored; the plaintext is returned once at
creation time and never again.

Table: ``api_keys`` (created by migration 0006).

Fields
------
id, tenant_id, name, key_hash (SHA-256 hex), scope, created_at,
last_used_at (nullable), revoked (bool default False)
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional, Tuple

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Session

from app.main import Base

# Key prefix so operators can identify Buro keys in logs.
KEY_PREFIX = "buro_"


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class ApiKey(Base):
    __tablename__ = "api_keys"

    id         = Column(Integer, primary_key=True, index=True)
    tenant_id  = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    name       = Column(String, nullable=False)          # human label
    key_hash   = Column(String, nullable=False, unique=True, index=True)
    scope      = Column(String, nullable=True)           # e.g. "ingest" | "read" | "admin"
    created_at = Column(DateTime(timezone=True), nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    revoked    = Column(Boolean, nullable=False, default=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def create_api_key(
    db: Session,
    tenant_id: int,
    name: str,
    scope: Optional[str] = None,
) -> Tuple[ApiKey, str]:
    """
    Create a new API key.

    Returns ``(ApiKey, raw_key)``.  The raw key is shown once and not stored.
    Row 44: key stored as SHA-256 hash only.
    """
    raw = KEY_PREFIX + secrets.token_hex(32)
    entry = ApiKey(
        tenant_id=tenant_id,
        name=name,
        key_hash=_hash(raw),
        scope=scope,
        created_at=datetime.now(timezone.utc),
        revoked=False,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry, raw


def lookup_api_key(db: Session, raw: str) -> Optional[ApiKey]:
    """
    Return a non-revoked ApiKey for *raw*, or None.
    Updates last_used_at on a hit.
    """
    entry = (
        db.query(ApiKey)
        .filter(ApiKey.key_hash == _hash(raw), ApiKey.revoked == False)  # noqa: E712
        .first()
    )
    if entry:
        entry.last_used_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(entry)
    return entry


def revoke_api_key(db: Session, key_id: int, tenant_id: int) -> bool:
    """Revoke a key.  Returns True if found and revoked, False otherwise."""
    entry = (
        db.query(ApiKey)
        .filter(ApiKey.id == key_id, ApiKey.tenant_id == tenant_id)
        .first()
    )
    if not entry:
        return False
    entry.revoked = True
    db.commit()
    return True


def list_api_keys(db: Session, tenant_id: int) -> list:
    """Return all non-revoked API keys for a tenant."""
    return (
        db.query(ApiKey)
        .filter(ApiKey.tenant_id == tenant_id, ApiKey.revoked == False)  # noqa: E712
        .order_by(ApiKey.created_at.desc())
        .all()
    )
