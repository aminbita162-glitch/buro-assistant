"""
app/domain/webhooks.py – signed outbound webhooks (row 45).

When a significant event occurs (message ingested, draft approved, etc.)
Buro can POST a JSON payload to the tenant's registered webhook URL.  The
payload is signed with HMAC-SHA256 using a per-subscription secret so the
receiver can verify authenticity.

Table: ``webhook_subscriptions`` (created by migration 0006).

Fields
------
id, tenant_id, url, secret_hash (SHA-256 of the plaintext secret),
events (comma-separated event filter, empty = all), active (bool),
created_at

Signing
-------
``X-Buro-Signature: sha256=<hex>``

The signature is HMAC-SHA256 of the raw JSON body bytes using the
plaintext secret.  The receiver computes the same digest and compares
using a constant-time equality check.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Session

from app.main import Base


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class WebhookSubscription(Base):
    __tablename__ = "webhook_subscriptions"

    id          = Column(Integer, primary_key=True, index=True)
    tenant_id   = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    url         = Column(String, nullable=False)
    # SHA-256 of the plaintext signing secret (never stored in plaintext).
    secret_hash = Column(String, nullable=False)
    # Comma-separated event filter; empty string means all events.
    events      = Column(String, nullable=False, default="")
    active      = Column(Boolean, nullable=False, default=True)
    created_at  = Column(DateTime(timezone=True), nullable=False)


# ---------------------------------------------------------------------------
# Signing helpers (row 45)
# ---------------------------------------------------------------------------

def _sign(payload_bytes: bytes, secret: str) -> str:
    """Return ``sha256=<hex>`` signature for *payload_bytes*."""
    sig = hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


def verify_signature(payload_bytes: bytes, secret: str, header_value: str) -> bool:
    """
    Constant-time comparison of *header_value* against the expected signature.
    Returns True when the signature is valid.
    """
    expected = _sign(payload_bytes, secret)
    return hmac.compare_digest(expected, header_value)


def build_signed_payload(event_type: str, data: Dict[str, Any], secret: str) -> Tuple[bytes, str]:
    """
    Serialise *data* to JSON and compute its HMAC-SHA256 signature.

    Returns ``(payload_bytes, signature_header_value)``.
    """
    body = json.dumps({"event": event_type, "data": data}, separators=(",", ":")).encode()
    sig = _sign(body, secret)
    return body, sig


# ---------------------------------------------------------------------------
# Subscription CRUD
# ---------------------------------------------------------------------------

def _hash_secret(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def create_subscription(
    db: Session,
    tenant_id: int,
    url: str,
    events: str = "",
) -> Tuple[WebhookSubscription, str]:
    """
    Register a webhook subscription.

    Returns ``(WebhookSubscription, plaintext_secret)``.
    The plaintext secret is shown once; only its hash is persisted.
    """
    raw_secret = secrets.token_hex(32)
    sub = WebhookSubscription(
        tenant_id=tenant_id,
        url=url,
        secret_hash=_hash_secret(raw_secret),
        events=events,
        active=True,
        created_at=datetime.now(timezone.utc),
    )
    db.add(sub)
    db.commit()
    db.refresh(sub)
    return sub, raw_secret


def active_subscriptions(
    db: Session,
    tenant_id: int,
    event_type: Optional[str] = None,
) -> List[WebhookSubscription]:
    """Return active subscriptions for *tenant_id*, optionally filtered by event."""
    rows = (
        db.query(WebhookSubscription)
        .filter(
            WebhookSubscription.tenant_id == tenant_id,
            WebhookSubscription.active == True,  # noqa: E712
        )
        .all()
    )
    if event_type:
        # Keep subscriptions that either have no event filter or explicitly include the event.
        rows = [
            r for r in rows
            if not r.events or event_type in r.events.split(",")
        ]
    return rows


def deactivate_subscription(db: Session, sub_id: int, tenant_id: int) -> bool:
    """Deactivate a subscription.  Returns True on success."""
    sub = (
        db.query(WebhookSubscription)
        .filter(
            WebhookSubscription.id == sub_id,
            WebhookSubscription.tenant_id == tenant_id,
        )
        .first()
    )
    if not sub:
        return False
    sub.active = False
    db.commit()
    return True
