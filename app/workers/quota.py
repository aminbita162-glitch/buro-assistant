"""
app/workers/quota.py – per-tenant daily quota (row 26).

Each tenant has a daily token budget.  The quota is checked before a model
call is placed.  If the tenant is over budget, the message is backpressured
(row 27 — handled in app/workers/queue.py via QuotaExceeded).

Quota config dict (stored in tenant policy or env):
{
  "daily_token_quota": 100000,   # total prompt+completion tokens per UTC day
  "queue_depth_cap": 500         # max pending items before backpressure
}

The ``quotas`` table (created by migration 0005) accumulates per-tenant
daily token usage.  A cron/worker resets it after UTC midnight.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import Column, Date, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Session

from app.main import Base


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class TenantQuota(Base):
    """One row per (tenant_id, quota_date).  Accumulates token usage."""
    __tablename__ = "quotas"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    quota_date = Column(Date, nullable=False, index=True)
    tokens_used = Column(Integer, nullable=False, default=0)
    cost_usd_used = Column(Float, nullable=False, default=0.0)


# ---------------------------------------------------------------------------
# Default limits
# ---------------------------------------------------------------------------

DEFAULT_DAILY_TOKEN_QUOTA = 100_000
DEFAULT_QUEUE_DEPTH_CAP   = 500


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class QuotaExceeded(Exception):
    """Raised when a tenant's daily token quota has been exhausted."""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def get_or_create_quota(db: Session, tenant_id: int, quota_date: Optional[date] = None) -> TenantQuota:
    """Return today's quota row, creating it if absent."""
    quota_date = quota_date or _today_utc()
    row = (
        db.query(TenantQuota)
        .filter(TenantQuota.tenant_id == tenant_id, TenantQuota.quota_date == quota_date)
        .first()
    )
    if row is None:
        row = TenantQuota(tenant_id=tenant_id, quota_date=quota_date,
                          tokens_used=0, cost_usd_used=0.0)
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def check_quota(
    db: Session,
    tenant_id: int,
    tokens_requested: int,
    policy_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Raise :class:`QuotaExceeded` if the tenant has insufficient daily budget.

    Row 26: per-tenant daily quota.
    """
    limit = int((policy_config or {}).get("daily_token_quota", DEFAULT_DAILY_TOKEN_QUOTA))
    row = get_or_create_quota(db, tenant_id)
    if row.tokens_used + tokens_requested > limit:
        raise QuotaExceeded(
            f"Tenant {tenant_id} daily quota {limit} exceeded "
            f"(used {row.tokens_used}, requested {tokens_requested})"
        )


def record_usage(
    db: Session,
    tenant_id: int,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
) -> TenantQuota:
    """Add token + cost usage to today's quota row."""
    row = get_or_create_quota(db, tenant_id)
    row.tokens_used  += tokens_in + tokens_out
    row.cost_usd_used += cost_usd
    db.commit()
    db.refresh(row)
    return row


def remaining_tokens(
    db: Session,
    tenant_id: int,
    policy_config: Optional[Dict[str, Any]] = None,
) -> int:
    """Return tokens remaining in today's budget (≥ 0)."""
    limit = int((policy_config or {}).get("daily_token_quota", DEFAULT_DAILY_TOKEN_QUOTA))
    row = get_or_create_quota(db, tenant_id)
    return max(0, limit - row.tokens_used)
