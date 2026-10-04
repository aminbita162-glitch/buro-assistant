"""
app/domain/plan_rules.py — Plan-level capability enforcement.

Commercial contract, Phase 3.

Rules (from DIRECTIVE.txt):
  desk    — cannot call Amin, Amilos, or Leila (no agents, no model).
  mail    — cannot call a model; agents that call a model are blocked.
  agents  — can call agents only while tokens_used < token_cap.
  trial   — send stays off; agents allowed under token cap.
  A rule hit costs zero tokens (the caller must not record tokens on a rule hit).

This module is pure Python.  It does not import SQLAlchemy models directly;
it accepts plan_code, status, tokens_used, and token_cap as plain values so
it can be called from any layer without a DB session.
"""
from __future__ import annotations

from typing import Optional

from app.domain.subscription import (
    PLAN_DESK,
    PLAN_MAIL,
    PLAN_AGENTS,
    PLAN_TRIAL,
    STATUS_TRIAL,
    STATUS_ACTIVE,
    STATUS_EXPIRED,
    STATUS_CANCELLED,
)

# ---------------------------------------------------------------------------
# Capability names — used as the *capability* argument to check_plan_capability
# ---------------------------------------------------------------------------

CAP_AGENT_AMIN   = "agent:amin"
CAP_AGENT_AMILOS = "agent:amilos"
CAP_AGENT_LEILA  = "agent:leila"
CAP_MODEL        = "model"
CAP_SEND         = "send"

# The three agent capabilities together (convenience set).
_AGENT_CAPS = frozenset([CAP_AGENT_AMIN, CAP_AGENT_AMILOS, CAP_AGENT_LEILA])


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class PlanRefusal:
    """Returned when a plan does not allow a capability."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __bool__(self) -> bool:          # falsy — a refusal is not allowed
        return False

    def __repr__(self) -> str:
        return f"PlanRefusal({self.reason!r})"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def check_plan_capability(
    capability: str,
    plan_code: str,
    status: str,
    tokens_used: int = 0,
    token_cap: int = 0,
) -> Optional[PlanRefusal]:
    """
    Return ``None`` when the capability is permitted, or a
    :class:`PlanRefusal` describing the reason for the refusal.

    Parameters
    ----------
    capability:
        One of the ``CAP_*`` constants defined in this module.
    plan_code:
        The tenant's plan code (trial / desk / mail / agents).
    status:
        The subscription status (trial / active / expired / cancelled).
    tokens_used:
        Current token usage for this billing period.
    token_cap:
        Maximum tokens allowed for this billing period.
    """
    # Expired and cancelled subscriptions are handled by can_route() in
    # subscription.py; they should not reach here, but guard defensively.
    if status in (STATUS_EXPIRED, STATUS_CANCELLED):
        return PlanRefusal("subscription_not_active")

    # -----------------------------------------------------------------------
    # Desk plan — agents and model are not included.
    # -----------------------------------------------------------------------
    if plan_code == PLAN_DESK:
        if capability in _AGENT_CAPS:
            return PlanRefusal("desk_plan_no_agents")
        if capability == CAP_MODEL:
            return PlanRefusal("desk_plan_no_model")

    # -----------------------------------------------------------------------
    # Mail plan — model is not included.
    # -----------------------------------------------------------------------
    elif plan_code == PLAN_MAIL:
        if capability == CAP_MODEL:
            return PlanRefusal("mail_plan_no_model")

    # -----------------------------------------------------------------------
    # Agents plan — agents and model allowed only under token cap.
    # -----------------------------------------------------------------------
    elif plan_code == PLAN_AGENTS:
        if capability in _AGENT_CAPS or capability == CAP_MODEL:
            if tokens_used >= token_cap:
                return PlanRefusal("agents_plan_token_cap_reached")

    # -----------------------------------------------------------------------
    # Trial — same token-cap rule as agents; send is additionally blocked.
    # -----------------------------------------------------------------------
    elif plan_code == PLAN_TRIAL:
        if capability in _AGENT_CAPS or capability == CAP_MODEL:
            if tokens_used >= token_cap:
                return PlanRefusal("trial_token_cap_reached")

    # Send is blocked on trial regardless of other rules.
    if capability == CAP_SEND and status == STATUS_TRIAL:
        return PlanRefusal("trial_send_blocked")

    return None   # capability permitted
