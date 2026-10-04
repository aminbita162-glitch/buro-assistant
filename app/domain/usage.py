"""
app/domain/usage.py – usage events (row 40).

A usage event is a lightweight, append-only record of a billable or
observable action: message ingested, model called, draft sent, API key
used, webhook fired.

ORM model: UsageEvent → table ``usage_events`` (created by migration 0006).

Fields
------
id, tenant_id, event_type, quantity (default 1), unit (tokens|messages|…),
cost_usd, actor (user email or "system" or api_key_id), reference_id
(opaque FK to the source row, e.g. decision id), created_at
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Session

from app.main import Base


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class UsageEvent(Base):
    __tablename__ = "usage_events"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)

    # Categorisation
    event_type   = Column(String, nullable=False, index=True)
    # e.g. message_ingested | model_called | draft_sent | api_key_used | webhook_fired

    # Quantity and unit for billing aggregation.
    quantity  = Column(Integer, nullable=False, default=1)
    unit      = Column(String,  nullable=True)    # "tokens" | "messages" | "webhooks" | …

    # Optional cost contribution.
    cost_usd  = Column(Float, nullable=True)

    # Who triggered it.
    actor     = Column(String, nullable=True)     # user email | "system" | api_key_id

    # Opaque reference to the triggering row (e.g. decision.id, message.id).
    reference_id = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def emit(
    db: Session,
    tenant_id: int,
    event_type: str,
    quantity: int = 1,
    unit: Optional[str] = None,
    cost_usd: Optional[float] = None,
    actor: str = "system",
    reference_id: Optional[int] = None,
) -> UsageEvent:
    """Append one usage event row.  Always INSERT — never UPDATE."""
    ev = UsageEvent(
        tenant_id=tenant_id,
        event_type=event_type,
        quantity=quantity,
        unit=unit,
        cost_usd=cost_usd,
        actor=actor,
        reference_id=reference_id,
        created_at=datetime.now(timezone.utc),
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return ev


def events_for_tenant(
    db: Session,
    tenant_id: int,
    event_type: Optional[str] = None,
    limit: int = 200,
) -> list:
    """Return recent usage events for a tenant, newest first."""
    q = (
        db.query(UsageEvent)
        .filter(UsageEvent.tenant_id == tenant_id)
    )
    if event_type:
        q = q.filter(UsageEvent.event_type == event_type)
    return q.order_by(UsageEvent.created_at.desc()).limit(limit).all()
