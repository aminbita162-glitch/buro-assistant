"""
tests/test_subscription.py — Commercial Phase 1: tenant subscription state tests.

Covers:
  - New tenant starts on trial
  - Trial state: can_route returns True within trial window
  - Expired state: can_route returns False after trial window closes
  - Active state: can_route returns True for paid plan
  - Cancelled state: can_route returns False
  - token_cap and tokens_used fields present and correct
  - record_tokens accumulates usage
  - table exists in schema

All tests run against in-memory SQLite from conftest.py.
No external service is contacted.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect as sa_inspect

from app.main import Base, engine, SessionLocal, Tenant

from app.domain.subscription import (
    TenantSubscription,
    TOKEN_CAP,
    PLAN_TRIAL,
    PLAN_DESK,
    PLAN_MAIL,
    PLAN_AGENTS,
    STATUS_TRIAL,
    STATUS_ACTIVE,
    STATUS_EXPIRED,
    STATUS_CANCELLED,
    TRIAL_DAYS,
    can_route,
    create_trial,
    get_subscription,
    record_tokens,
    refresh_status,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def _make_tenant(slug: str = "t1") -> Tenant:
    db = SessionLocal()
    try:
        t = Tenant(name=slug, slug=slug)
        db.add(t)
        db.commit()
        db.refresh(t)
        return t
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSubscriptionSchema:
    def test_table_exists(self):
        insp = sa_inspect(engine)
        assert "tenant_subscriptions" in insp.get_table_names()

    def test_required_columns_present(self):
        insp = sa_inspect(engine)
        cols = {c["name"] for c in insp.get_columns("tenant_subscriptions")}
        for col in ("id", "tenant_id", "plan_code", "status", "trial_end",
                    "token_cap", "tokens_used"):
            assert col in cols, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# State 1 — trial
# ---------------------------------------------------------------------------


class TestTrialState:
    def test_new_tenant_gets_trial(self):
        tenant = _make_tenant("sub-trial1")
        db = SessionLocal()
        try:
            sub = create_trial(db, tenant.id)
            assert sub.status == STATUS_TRIAL
            assert sub.plan_code == PLAN_TRIAL
        finally:
            db.close()

    def test_trial_end_is_10_days_from_now(self):
        tenant = _make_tenant("sub-trial2")
        db = SessionLocal()
        try:
            sub = create_trial(db, tenant.id)
            expected = datetime.now(timezone.utc) + timedelta(days=TRIAL_DAYS)
            delta = abs((sub.trial_end.replace(tzinfo=timezone.utc) - expected).total_seconds())
            assert delta < 5  # within 5 seconds
        finally:
            db.close()

    def test_trial_token_cap(self):
        tenant = _make_tenant("sub-trial3")
        db = SessionLocal()
        try:
            sub = create_trial(db, tenant.id)
            assert sub.token_cap == TOKEN_CAP[PLAN_TRIAL]
        finally:
            db.close()

    def test_trial_tokens_used_starts_at_zero(self):
        tenant = _make_tenant("sub-trial4")
        db = SessionLocal()
        try:
            sub = create_trial(db, tenant.id)
            assert sub.tokens_used == 0
        finally:
            db.close()

    def test_can_route_during_trial(self):
        tenant = _make_tenant("sub-trial5")
        db = SessionLocal()
        try:
            create_trial(db, tenant.id)
            assert can_route(db, tenant.id) is True
        finally:
            db.close()


# ---------------------------------------------------------------------------
# State 2 — expired (trial window closed)
# ---------------------------------------------------------------------------


class TestExpiredState:
    def _make_expired_subscription(self, db, tenant_id: int) -> TenantSubscription:
        """Insert a trial subscription whose trial_end is in the past."""
        past = datetime.now(timezone.utc) - timedelta(days=1)
        sub = TenantSubscription(
            tenant_id=tenant_id,
            plan_code=PLAN_TRIAL,
            status=STATUS_TRIAL,
            trial_end=past,
            token_cap=TOKEN_CAP[PLAN_TRIAL],
            tokens_used=0,
        )
        db.add(sub)
        db.commit()
        db.refresh(sub)
        return sub

    def test_refresh_status_advances_expired_trial(self):
        tenant = _make_tenant("sub-exp1")
        db = SessionLocal()
        try:
            sub = self._make_expired_subscription(db, tenant.id)
            sub = refresh_status(db, sub)
            assert sub.status == STATUS_EXPIRED
        finally:
            db.close()

    def test_can_route_after_trial_expired(self):
        tenant = _make_tenant("sub-exp2")
        db = SessionLocal()
        try:
            self._make_expired_subscription(db, tenant.id)
            assert can_route(db, tenant.id) is False
        finally:
            db.close()

    def test_can_route_no_subscription_returns_false(self):
        tenant = _make_tenant("sub-exp3")
        db = SessionLocal()
        try:
            assert can_route(db, tenant.id) is False
        finally:
            db.close()


# ---------------------------------------------------------------------------
# State 3 — active (paid plan)
# ---------------------------------------------------------------------------


class TestActiveState:
    def _make_active_subscription(
        self, db, tenant_id: int, plan: str = PLAN_DESK
    ) -> TenantSubscription:
        sub = TenantSubscription(
            tenant_id=tenant_id,
            plan_code=plan,
            status=STATUS_ACTIVE,
            trial_end=None,
            token_cap=TOKEN_CAP[plan],
            tokens_used=0,
        )
        db.add(sub)
        db.commit()
        db.refresh(sub)
        return sub

    def test_active_desk_can_route(self):
        tenant = _make_tenant("sub-act1")
        db = SessionLocal()
        try:
            self._make_active_subscription(db, tenant.id, PLAN_DESK)
            assert can_route(db, tenant.id) is True
        finally:
            db.close()

    def test_active_mail_can_route(self):
        tenant = _make_tenant("sub-act2")
        db = SessionLocal()
        try:
            self._make_active_subscription(db, tenant.id, PLAN_MAIL)
            assert can_route(db, tenant.id) is True
        finally:
            db.close()

    def test_active_agents_can_route(self):
        tenant = _make_tenant("sub-act3")
        db = SessionLocal()
        try:
            self._make_active_subscription(db, tenant.id, PLAN_AGENTS)
            assert can_route(db, tenant.id) is True
        finally:
            db.close()

    def test_active_status_not_changed_by_refresh(self):
        tenant = _make_tenant("sub-act4")
        db = SessionLocal()
        try:
            sub = self._make_active_subscription(db, tenant.id, PLAN_DESK)
            sub = refresh_status(db, sub)
            assert sub.status == STATUS_ACTIVE
        finally:
            db.close()

    def test_desk_token_cap(self):
        tenant = _make_tenant("sub-act5")
        db = SessionLocal()
        try:
            sub = self._make_active_subscription(db, tenant.id, PLAN_DESK)
            assert sub.token_cap == TOKEN_CAP[PLAN_DESK]
        finally:
            db.close()


# ---------------------------------------------------------------------------
# State 4 — cancelled
# ---------------------------------------------------------------------------


class TestCancelledState:
    def _make_cancelled_subscription(self, db, tenant_id: int) -> TenantSubscription:
        sub = TenantSubscription(
            tenant_id=tenant_id,
            plan_code=PLAN_DESK,
            status=STATUS_CANCELLED,
            trial_end=None,
            token_cap=TOKEN_CAP[PLAN_DESK],
            tokens_used=0,
        )
        db.add(sub)
        db.commit()
        db.refresh(sub)
        return sub

    def test_cancelled_cannot_route(self):
        tenant = _make_tenant("sub-can1")
        db = SessionLocal()
        try:
            self._make_cancelled_subscription(db, tenant.id)
            assert can_route(db, tenant.id) is False
        finally:
            db.close()

    def test_cancelled_status_not_changed_by_refresh(self):
        tenant = _make_tenant("sub-can2")
        db = SessionLocal()
        try:
            sub = self._make_cancelled_subscription(db, tenant.id)
            sub = refresh_status(db, sub)
            assert sub.status == STATUS_CANCELLED
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


class TestTokenAccounting:
    def test_record_tokens_accumulates(self):
        tenant = _make_tenant("sub-tok1")
        db = SessionLocal()
        try:
            create_trial(db, tenant.id)
            record_tokens(db, tenant.id, 100)
            sub = get_subscription(db, tenant.id)
            assert sub.tokens_used == 100
        finally:
            db.close()

    def test_record_tokens_multiple_calls(self):
        tenant = _make_tenant("sub-tok2")
        db = SessionLocal()
        try:
            create_trial(db, tenant.id)
            record_tokens(db, tenant.id, 50)
            record_tokens(db, tenant.id, 75)
            sub = get_subscription(db, tenant.id)
            assert sub.tokens_used == 125
        finally:
            db.close()

    def test_get_subscription_returns_none_for_unknown_tenant(self):
        db = SessionLocal()
        try:
            result = get_subscription(db, 999999)
            assert result is None
        finally:
            db.close()
