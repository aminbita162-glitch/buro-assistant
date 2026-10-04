"""
app/domain/subscription.py — Tenant subscription record.

Commercial contract, Phase 1.

A tenant subscription has four states:
  trial     — within 10-day trial window, token cap enforced, send blocked
  active    — paid plan, routes open, token cap enforced
  expired   — trial ended or plan lapsed, agent routes refuse work
  cancelled — operator-cancelled, agent routes refuse work

Plan codes (from DIRECTIVE.txt):
  trial     — auto-assigned on new tenant
  desk      — Plan 1 (65 EUR/mo)
  mail      — Plan 2 (149 EUR/mo)
  agents    — Plan 3 (270 EUR/mo)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Session

from app.main import Base

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRIAL_DAYS = 10

# Default token cap per plan code.
TOKEN_CAP: dict[str, int] = {
    "trial": 50_000,
    "desk": 500_000,
    "mail": 1_000_000,
    "agents": 3_000_000,
}

# Status constants — the four states.
STATUS_TRIAL = "trial"
STATUS_ACTIVE = "active"
STATUS_EXPIRED = "expired"
STATUS_CANCELLED = "cancelled"

# Plan code constants.
PLAN_TRIAL = "trial"
PLAN_DESK = "desk"
PLAN_MAIL = "mail"
PLAN_AGENTS = "agents"


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class TenantSubscription(Base):
    __tablename__ = "tenant_subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(
        Integer, ForeignKey("tenants.id"), nullable=False, unique=True, index=True
    )
    plan_code = Column(String, nullable=False, default=PLAN_TRIAL)
    status = Column(String, nullable=False, default=STATUS_TRIAL)
    trial_end = Column(DateTime(timezone=True), nullable=True)
    token_cap = Column(Integer, nullable=False, default=TOKEN_CAP[PLAN_TRIAL])
    tokens_used = Column(Integer, nullable=False, default=0)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def create_trial(db: Session, tenant_id: int) -> TenantSubscription:
    """Create a new trial subscription for a tenant.  Called at tenant creation."""
    trial_end = datetime.now(timezone.utc) + timedelta(days=TRIAL_DAYS)
    sub = TenantSubscription(
        tenant_id=tenant_id,
        plan_code=PLAN_TRIAL,
        status=STATUS_TRIAL,
        trial_end=trial_end,
        token_cap=TOKEN_CAP[PLAN_TRIAL],
        tokens_used=0,
    )
    db.add(sub)
    db.commit()
    db.refresh(sub)
    return sub


def get_subscription(db: Session, tenant_id: int) -> Optional[TenantSubscription]:
    """Return the subscription for a tenant, or None if not found."""
    return (
        db.query(TenantSubscription)
        .filter(TenantSubscription.tenant_id == tenant_id)
        .first()
    )


def refresh_status(db: Session, sub: TenantSubscription) -> TenantSubscription:
    """
    Advance a trial subscription to expired when the trial window has closed.
    Active/expired/cancelled subscriptions are not changed.
    """
    if sub.status == STATUS_TRIAL and sub.trial_end is not None:
        now = datetime.now(timezone.utc)
        trial_end = sub.trial_end
        if trial_end.tzinfo is None:
            trial_end = trial_end.replace(tzinfo=timezone.utc)
        if now > trial_end:
            sub.status = STATUS_EXPIRED
            db.commit()
            db.refresh(sub)
    return sub


def can_route(db: Session, tenant_id: int) -> bool:
    """
    Return True when the tenant's agent routes may accept work.

    Trial and active subscriptions allow routing (subject to token cap).
    Expired and cancelled subscriptions return False.
    """
    sub = get_subscription(db, tenant_id)
    if sub is None:
        return False
    sub = refresh_status(db, sub)
    return sub.status in (STATUS_TRIAL, STATUS_ACTIVE)


def record_tokens(db: Session, tenant_id: int, count: int) -> TenantSubscription:
    """Add *count* to the tenant's tokens_used counter."""
    sub = get_subscription(db, tenant_id)
    if sub is None:
        raise ValueError(f"No subscription for tenant {tenant_id}")
    sub.tokens_used = (sub.tokens_used or 0) + count
    db.commit()
    db.refresh(sub)
    return sub
