"""
app/workers/webhook_delivery.py – signed webhook delivery with retries
and a delivery log (follow-up Phase 2).

Delivery flow
-------------
1. Look up active subscriptions for the tenant and event type.
2. Build the signed payload using :func:`~app.domain.webhooks.build_signed_payload`.
3. POST to the subscription URL with the ``X-Buro-Signature`` header.
4. Retry up to ``max_attempts`` times with exponential back-off (capped at
   ``max_delay_seconds``).
5. Write one :class:`DeliveryLog` row per attempt (success or failure).

The signing secret is stored as a hash in the DB; the plaintext secret is
passed by the caller (retrieved at subscription creation time and kept in the
environment or secrets store — never in the DB).

Since the plaintext secret is not persisted, delivery is driven by the caller
who holds the secret.  For testing, :func:`deliver_event` accepts an explicit
``secret`` argument.

Secrets
-------
No secret is logged or included in any DeliveryLog row.  ``payload_preview``
is the first 200 bytes of the JSON body (no auth material there).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Session

from app.domain.webhooks import (
    WebhookSubscription,
    active_subscriptions,
    build_signed_payload,
)
from app.main import Base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_ATTEMPTS: int = 3
DEFAULT_BASE_DELAY: float = 1.0   # seconds between retries (doubled each attempt)
DEFAULT_MAX_DELAY: float = 60.0   # cap on retry delay
REQUEST_TIMEOUT: float = 10.0     # seconds per HTTP request


# ---------------------------------------------------------------------------
# ORM: delivery_log
# ---------------------------------------------------------------------------

class DeliveryLog(Base):
    """
    One row per delivery attempt.

    ``status``: "ok" | "error" | "timeout"
    ``http_status``: HTTP response code (None when no response was received).
    ``payload_preview``: first 200 bytes of the JSON body (no auth material).
    """
    __tablename__ = "delivery_log"

    id               = Column(Integer, primary_key=True, index=True)
    tenant_id        = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    subscription_id  = Column(Integer, ForeignKey("webhook_subscriptions.id"),
                              nullable=False, index=True)
    event_type       = Column(String, nullable=False)
    attempt          = Column(Integer, nullable=False, default=1)
    status           = Column(String, nullable=False, default="ok")   # ok | error | timeout
    http_status      = Column(Integer, nullable=True)
    error_detail     = Column(Text, nullable=True)
    payload_preview  = Column(Text, nullable=True)
    delivered_at     = Column(DateTime(timezone=True), nullable=False)
    success          = Column(Boolean, nullable=False, default=False)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DeliveryResult:
    """Outcome of one :func:`deliver_event` call."""
    subscription_id: int
    success: bool
    attempts: int
    last_http_status: Optional[int]
    last_error: Optional[str]


# ---------------------------------------------------------------------------
# Internal: single HTTP attempt
# ---------------------------------------------------------------------------

def _attempt_post(
    url: str,
    payload_bytes: bytes,
    signature: str,
) -> tuple[bool, int | None, str | None]:
    """
    POST *payload_bytes* to *url*.

    Returns ``(success, http_status, error_detail)``.
    Never raises.
    """
    try:
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            url,
            data=payload_bytes,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Buro-Signature": signature,
            },
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            http_status: int = resp.status
            return (200 <= http_status < 300), http_status, None
    except urllib.error.HTTPError as exc:
        return False, exc.code, f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)


# ---------------------------------------------------------------------------
# Public: deliver one event to one subscription
# ---------------------------------------------------------------------------

def deliver_event(
    db: Session,
    tenant_id: int,
    subscription: "WebhookSubscription",
    event_type: str,
    data: Dict[str, Any],
    secret: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> DeliveryResult:
    """
    Deliver *event_type* with *data* to *subscription*, retrying on failure.

    Parameters
    ----------
    secret:
        Plaintext signing secret for this subscription.  Never stored or logged.
    """
    payload_bytes, signature = build_signed_payload(event_type, data, secret)
    preview = payload_bytes[:200].decode("utf-8", errors="replace")

    last_http: Optional[int] = None
    last_err: Optional[str] = None
    success = False

    delay = base_delay
    for attempt in range(1, max_attempts + 1):
        ok, http_status, error_detail = _attempt_post(
            subscription.url, payload_bytes, signature
        )
        last_http = http_status
        last_err = error_detail

        log_status = "ok" if ok else ("timeout" if http_status is None else "error")

        _write_log(
            db,
            tenant_id=tenant_id,
            subscription_id=subscription.id,
            event_type=event_type,
            attempt=attempt,
            status=log_status,
            http_status=http_status,
            error_detail=error_detail,
            payload_preview=preview,
            success=ok,
        )

        if ok:
            success = True
            break

        if attempt < max_attempts:
            time.sleep(min(delay, max_delay))
            delay *= 2

        logger.warning(
            "webhook_delivery: attempt %d/%d failed for sub=%d event=%s: %s",
            attempt, max_attempts, subscription.id, event_type, error_detail,
        )

    if not success:
        logger.error(
            "webhook_delivery: all %d attempts failed for sub=%d event=%s",
            max_attempts, subscription.id, event_type,
        )

    return DeliveryResult(
        subscription_id=subscription.id,
        success=success,
        attempts=attempt,
        last_http_status=last_http,
        last_error=last_err,
    )


# ---------------------------------------------------------------------------
# Public: deliver to all subscriptions for a tenant event
# ---------------------------------------------------------------------------

def dispatch_event(
    db: Session,
    tenant_id: int,
    event_type: str,
    data: Dict[str, Any],
    secrets_map: Dict[int, str],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> List[DeliveryResult]:
    """
    Deliver *event_type* to every active subscription for *tenant_id*.

    Parameters
    ----------
    secrets_map:
        Mapping from subscription id → plaintext secret.
        Subscriptions whose id is absent from the map are skipped silently
        (the secret is not available to this process).
    """
    subs = active_subscriptions(db, tenant_id, event_type=event_type)
    results: List[DeliveryResult] = []
    for sub in subs:
        secret = secrets_map.get(sub.id)
        if not secret:
            logger.warning(
                "dispatch_event: no secret for sub=%d; skipped", sub.id
            )
            continue
        result = deliver_event(
            db,
            tenant_id=tenant_id,
            subscription=sub,
            event_type=event_type,
            data=data,
            secret=secret,
            max_attempts=max_attempts,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Internal: write one DeliveryLog row
# ---------------------------------------------------------------------------

def _write_log(
    db: Session,
    tenant_id: int,
    subscription_id: int,
    event_type: str,
    attempt: int,
    status: str,
    http_status: Optional[int],
    error_detail: Optional[str],
    payload_preview: str,
    success: bool,
) -> None:
    row = DeliveryLog(
        tenant_id=tenant_id,
        subscription_id=subscription_id,
        event_type=event_type,
        attempt=attempt,
        status=status,
        http_status=http_status,
        error_detail=error_detail,
        payload_preview=payload_preview,
        delivered_at=datetime.now(timezone.utc),
        success=success,
    )
    try:
        db.add(row)
        db.commit()
    except Exception:  # noqa: BLE001
        db.rollback()
