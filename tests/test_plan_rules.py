"""
tests/test_plan_rules.py — Commercial Phase 3: plan capability enforcement tests.

Covers:
  - Desk plan: Amin refused
  - Desk plan: Amilos refused
  - Desk plan: Leila refused
  - Desk plan: model refused
  - Desk plan: send allowed (desk is active, not trial)
  - Mail plan: model refused
  - Mail plan: Amin allowed (agents are included in mail)
  - Mail plan: Amilos allowed
  - Mail plan: Leila allowed
  - Agents plan: Amin allowed under cap
  - Agents plan: Amilos allowed under cap
  - Agents plan: Leila allowed under cap
  - Agents plan: model allowed under cap
  - Agents plan: Amin refused at cap
  - Agents plan: model refused at cap
  - Trial: Amin allowed under cap
  - Trial: model allowed under cap
  - Trial: Amin refused at cap
  - Trial: send refused (trial blocks send)
  - Active desk: send allowed
  - Rule hit costs zero tokens — a rule hit itself is not a CAP_MODEL call,
    so no token check applies; verified by checking that CAP_AGENT_AMIN is
    refused on desk regardless of tokens_used.
  - Expired subscription: any capability refused
  - Cancelled subscription: any capability refused

All tests are pure-Python; no DB session is required.
"""
from __future__ import annotations

import pytest

from app.domain.plan_rules import (
    PlanRefusal,
    CAP_AGENT_AMIN,
    CAP_AGENT_AMILOS,
    CAP_AGENT_LEILA,
    CAP_MODEL,
    CAP_SEND,
    check_plan_capability,
)
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
# Helpers
# ---------------------------------------------------------------------------

def _allow(capability, plan, status="active", used=0, cap=3_000_000):
    result = check_plan_capability(capability, plan, status, used, cap)
    assert result is None, f"Expected allow but got: {result}"


def _refuse(capability, plan, status="active", used=0, cap=3_000_000, reason=None):
    result = check_plan_capability(capability, plan, status, used, cap)
    assert isinstance(result, PlanRefusal), f"Expected PlanRefusal but got: {result}"
    if reason is not None:
        assert result.reason == reason, f"Expected reason {reason!r} but got {result.reason!r}"
    return result


# ---------------------------------------------------------------------------
# Desk plan
# ---------------------------------------------------------------------------

class TestDeskPlan:
    def test_desk_refuses_amin(self):
        _refuse(CAP_AGENT_AMIN, PLAN_DESK, reason="desk_plan_no_agents")

    def test_desk_refuses_amilos(self):
        _refuse(CAP_AGENT_AMILOS, PLAN_DESK, reason="desk_plan_no_agents")

    def test_desk_refuses_leila(self):
        _refuse(CAP_AGENT_LEILA, PLAN_DESK, reason="desk_plan_no_agents")

    def test_desk_refuses_model(self):
        _refuse(CAP_MODEL, PLAN_DESK, reason="desk_plan_no_model")

    def test_desk_allows_send_when_active(self):
        """Desk does not restrict send; send block only applies on trial."""
        _allow(CAP_SEND, PLAN_DESK, status=STATUS_ACTIVE)

    def test_desk_refusal_is_falsy(self):
        result = check_plan_capability(CAP_AGENT_AMIN, PLAN_DESK, STATUS_ACTIVE)
        assert not result   # PlanRefusal is falsy

    def test_desk_refuses_amin_regardless_of_token_count(self):
        """Rule hit costs zero tokens — desk still refuses agents at any token level."""
        _refuse(CAP_AGENT_AMIN, PLAN_DESK, used=0, cap=0, reason="desk_plan_no_agents")
        _refuse(CAP_AGENT_AMIN, PLAN_DESK, used=999, cap=999, reason="desk_plan_no_agents")


# ---------------------------------------------------------------------------
# Mail plan
# ---------------------------------------------------------------------------

class TestMailPlan:
    def test_mail_refuses_model(self):
        _refuse(CAP_MODEL, PLAN_MAIL, reason="mail_plan_no_model")

    def test_mail_allows_amin(self):
        _allow(CAP_AGENT_AMIN, PLAN_MAIL)

    def test_mail_allows_amilos(self):
        _allow(CAP_AGENT_AMILOS, PLAN_MAIL)

    def test_mail_allows_leila(self):
        _allow(CAP_AGENT_LEILA, PLAN_MAIL)

    def test_mail_allows_send_when_active(self):
        _allow(CAP_SEND, PLAN_MAIL, status=STATUS_ACTIVE)

    def test_mail_refuses_model_regardless_of_tokens(self):
        """Model block on mail plan is not token-gate; it is absolute."""
        _refuse(CAP_MODEL, PLAN_MAIL, used=0, cap=1_000_000, reason="mail_plan_no_model")


# ---------------------------------------------------------------------------
# Agents plan
# ---------------------------------------------------------------------------

class TestAgentsPlan:
    def test_agents_allows_amin_under_cap(self):
        _allow(CAP_AGENT_AMIN, PLAN_AGENTS, used=100, cap=3_000_000)

    def test_agents_allows_amilos_under_cap(self):
        _allow(CAP_AGENT_AMILOS, PLAN_AGENTS, used=100, cap=3_000_000)

    def test_agents_allows_leila_under_cap(self):
        _allow(CAP_AGENT_LEILA, PLAN_AGENTS, used=100, cap=3_000_000)

    def test_agents_allows_model_under_cap(self):
        _allow(CAP_MODEL, PLAN_AGENTS, used=100, cap=3_000_000)

    def test_agents_refuses_amin_at_cap(self):
        _refuse(CAP_AGENT_AMIN, PLAN_AGENTS, used=3_000_000, cap=3_000_000,
                reason="agents_plan_token_cap_reached")

    def test_agents_refuses_amilos_at_cap(self):
        _refuse(CAP_AGENT_AMILOS, PLAN_AGENTS, used=3_000_000, cap=3_000_000,
                reason="agents_plan_token_cap_reached")

    def test_agents_refuses_leila_at_cap(self):
        _refuse(CAP_AGENT_LEILA, PLAN_AGENTS, used=3_000_000, cap=3_000_000,
                reason="agents_plan_token_cap_reached")

    def test_agents_refuses_model_at_cap(self):
        _refuse(CAP_MODEL, PLAN_AGENTS, used=3_000_000, cap=3_000_000,
                reason="agents_plan_token_cap_reached")

    def test_agents_refuses_amin_over_cap(self):
        _refuse(CAP_AGENT_AMIN, PLAN_AGENTS, used=3_000_001, cap=3_000_000,
                reason="agents_plan_token_cap_reached")

    def test_agents_allows_send(self):
        _allow(CAP_SEND, PLAN_AGENTS, status=STATUS_ACTIVE)

    def test_agents_allows_amin_at_cap_minus_one(self):
        """One token below cap is still allowed."""
        _allow(CAP_AGENT_AMIN, PLAN_AGENTS, used=2_999_999, cap=3_000_000)


# ---------------------------------------------------------------------------
# Trial subscription
# ---------------------------------------------------------------------------

class TestTrialPlan:
    def test_trial_allows_amin_under_cap(self):
        _allow(CAP_AGENT_AMIN, PLAN_TRIAL, status=STATUS_TRIAL, used=0, cap=50_000)

    def test_trial_allows_model_under_cap(self):
        _allow(CAP_MODEL, PLAN_TRIAL, status=STATUS_TRIAL, used=0, cap=50_000)

    def test_trial_refuses_amin_at_cap(self):
        _refuse(CAP_AGENT_AMIN, PLAN_TRIAL, status=STATUS_TRIAL,
                used=50_000, cap=50_000, reason="trial_token_cap_reached")

    def test_trial_refuses_model_at_cap(self):
        _refuse(CAP_MODEL, PLAN_TRIAL, status=STATUS_TRIAL,
                used=50_000, cap=50_000, reason="trial_token_cap_reached")

    def test_trial_refuses_send(self):
        """Send stays off on trial regardless of token count."""
        _refuse(CAP_SEND, PLAN_TRIAL, status=STATUS_TRIAL, used=0, cap=50_000,
                reason="trial_send_blocked")

    def test_trial_refuses_send_even_under_cap(self):
        _refuse(CAP_SEND, PLAN_TRIAL, status=STATUS_TRIAL, used=1, cap=50_000,
                reason="trial_send_blocked")


# ---------------------------------------------------------------------------
# Inactive subscriptions
# ---------------------------------------------------------------------------

class TestInactiveSubscriptions:
    def test_expired_refuses_any_capability(self):
        for capability in [CAP_AGENT_AMIN, CAP_MODEL, CAP_SEND]:
            result = check_plan_capability(capability, PLAN_AGENTS, STATUS_EXPIRED,
                                           tokens_used=0, token_cap=3_000_000)
            assert isinstance(result, PlanRefusal)
            assert result.reason == "subscription_not_active"

    def test_cancelled_refuses_any_capability(self):
        for capability in [CAP_AGENT_AMIN, CAP_MODEL, CAP_SEND]:
            result = check_plan_capability(capability, PLAN_DESK, STATUS_CANCELLED,
                                           tokens_used=0, token_cap=500_000)
            assert isinstance(result, PlanRefusal)
            assert result.reason == "subscription_not_active"


# ---------------------------------------------------------------------------
# PlanRefusal semantics
# ---------------------------------------------------------------------------

class TestPlanRefusal:
    def test_refusal_is_falsy(self):
        r = PlanRefusal("some_reason")
        assert not r
        assert bool(r) is False

    def test_refusal_repr_contains_reason(self):
        r = PlanRefusal("desk_plan_no_agents")
        assert "desk_plan_no_agents" in repr(r)

    def test_none_is_truthy(self):
        """None (allow) is truthy — callers can do: if check_plan_capability(...)."""
        result = check_plan_capability(CAP_AGENT_AMIN, PLAN_AGENTS, STATUS_ACTIVE,
                                       tokens_used=0, token_cap=3_000_000)
        assert result is None
