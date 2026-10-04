"""
tests/test_operator_surface.py — Commercial Phase 4: operator surface tests.

Covers:
  - GET /desk/plan returns 401 without token
  - GET /desk/plan returns 200 with valid token
  - plan response has required keys: plan_code, status, trial_days_left,
    token_cap, tokens_used, choose_plan, plans_available
  - no card field in any response key
  - choose_plan is False for trial and active
  - choose_plan is True for expired subscription
  - choose_plan is True for cancelled subscription
  - trial_days_left is positive integer for live trial
  - trial_days_left is 0 for non-trial plan
  - plans_available contains three entries (desk, mail, agents)
  - plans_available entries have code, name, price_eur_month, description; no card field
  - no subscription row → choose_plan True
  - Pipeline: desk plan refuses CAP_AGENT_AMIN → outcome="plan_refused"
  - Pipeline: desk plan refuses CAP_AGENT_AMILOS on draft_reply path
  - Pipeline: mail plan refuses CAP_MODEL when model is passed
  - Pipeline: agents plan at token cap refuses agent call
  - Pipeline: agents plan under cap allows agent call (outcome != plan_refused)
  - Pipeline: trial plan under cap allows agent call
  - Pipeline: expired status refuses any agent call

All tests use the in-memory SQLite engine from conftest.py.
No external service is contacted.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.main import (
    app, Base, engine, SessionLocal,
    Tenant, User,
    hash_password, limiter,
)
from app.domain.subscription import (
    TenantSubscription,
    TOKEN_CAP,
    PLAN_TRIAL, PLAN_DESK, PLAN_MAIL, PLAN_AGENTS,
    STATUS_TRIAL, STATUS_ACTIVE, STATUS_EXPIRED, STATUS_CANCELLED,
    create_trial,
)
from app.pipeline import run_pipeline, PipelineResult
from app.agents.fake_model import FakeModel
from app.ingest.normalize import NormalizedMessage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    try:
        limiter._storage.reset()
    except Exception:
        pass
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tenant(slug: str = "t1") -> Tenant:
    db = SessionLocal()
    try:
        t = Tenant(name=slug, slug=slug)
        db.add(t); db.commit(); db.refresh(t)
        return t
    finally:
        db.close()


def _make_user(tenant: Tenant, email: str = "op@example.com") -> User:
    db = SessionLocal()
    try:
        u = User(
            tenant_id=tenant.id, name="Operator",
            email=email, password_hash=hash_password("pass99"),
        )
        db.add(u); db.commit(); db.refresh(u)
        return u
    finally:
        db.close()


def _login(client, email: str, password: str = "pass99") -> str:
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def _make_subscription(
    tenant_id: int,
    plan_code: str = PLAN_TRIAL,
    status: str = STATUS_TRIAL,
    trial_end=None,
    token_cap: int = 50_000,
    tokens_used: int = 0,
) -> TenantSubscription:
    db = SessionLocal()
    try:
        if trial_end is None and status == STATUS_TRIAL:
            trial_end = datetime.now(timezone.utc) + timedelta(days=10)
        sub = TenantSubscription(
            tenant_id=tenant_id,
            plan_code=plan_code,
            status=status,
            trial_end=trial_end,
            token_cap=token_cap,
            tokens_used=tokens_used,
        )
        db.add(sub); db.commit(); db.refresh(sub)
        return sub
    finally:
        db.close()


def _msg(
    subject: str = "Invoice #100",
    body: str = "Please process the attached invoice.",
    provider_message_id: str = "msg-001",
    tenant_id: int = 1,
) -> NormalizedMessage:
    return NormalizedMessage(
        provider="fake",
        provider_message_id=provider_message_id,
        tenant_id=tenant_id,
        message_id_header=None,
        subject=subject,
        subject_normalized=NormalizedMessage.normalize_subject(subject),
        sender="user@example.com",
        recipients=["desk@co.com"],
        body_text=body,
        attachments=[],
        raw={},
    )


_RULE_PACK_INVOICE = {
    "domain_rules": [],
    "subject_rules": [
        {"match": "invoice", "department": "billing", "action": "draft_reply"},
    ],
    "department_rules": [],
    "confidence_threshold": 0.7,
}

_RULE_PACK_HOLD = {
    "domain_rules": [],
    "subject_rules": [
        {"match": "hold", "department": "general", "action": "hold"},
    ],
    "department_rules": [],
    "confidence_threshold": 0.7,
}


# ---------------------------------------------------------------------------
# /desk/plan — auth guard
# ---------------------------------------------------------------------------

class TestDeskPlanAuth:
    def test_plan_requires_auth(self, client):
        r = client.get("/desk/plan")
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# /desk/plan — response shape
# ---------------------------------------------------------------------------

class TestDeskPlanShape:
    def test_plan_returns_200(self, client):
        tenant = _make_tenant("sp1")
        _make_user(tenant)
        _make_subscription(tenant.id)
        token = _login(client, "op@example.com")
        r = client.get("/desk/plan", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_plan_has_required_keys(self, client):
        tenant = _make_tenant("sp2")
        _make_user(tenant)
        _make_subscription(tenant.id)
        token = _login(client, "op@example.com")
        data = client.get(
            "/desk/plan", headers={"Authorization": f"Bearer {token}"}
        ).json()
        for key in ("plan_code", "status", "trial_days_left", "token_cap",
                    "tokens_used", "choose_plan", "plans_available"):
            assert key in data, f"missing key: {key}"

    def test_plan_response_has_no_card_field(self, client):
        tenant = _make_tenant("sp3")
        _make_user(tenant)
        _make_subscription(tenant.id)
        token = _login(client, "op@example.com")
        raw = client.get(
            "/desk/plan", headers={"Authorization": f"Bearer {token}"}
        ).text
        assert "card" not in raw.lower()

    def test_plans_available_has_three_entries(self, client):
        tenant = _make_tenant("sp4")
        _make_user(tenant)
        _make_subscription(tenant.id)
        token = _login(client, "op@example.com")
        data = client.get(
            "/desk/plan", headers={"Authorization": f"Bearer {token}"}
        ).json()
        assert len(data["plans_available"]) == 3

    def test_plans_available_codes(self, client):
        tenant = _make_tenant("sp5")
        _make_user(tenant)
        _make_subscription(tenant.id)
        token = _login(client, "op@example.com")
        data = client.get(
            "/desk/plan", headers={"Authorization": f"Bearer {token}"}
        ).json()
        codes = {p["code"] for p in data["plans_available"]}
        assert codes == {"desk", "mail", "agents"}

    def test_plan_entry_has_no_card_field(self, client):
        tenant = _make_tenant("sp6")
        _make_user(tenant)
        _make_subscription(tenant.id)
        token = _login(client, "op@example.com")
        data = client.get(
            "/desk/plan", headers={"Authorization": f"Bearer {token}"}
        ).json()
        for entry in data["plans_available"]:
            assert "card" not in entry


# ---------------------------------------------------------------------------
# /desk/plan — choose_plan state
# ---------------------------------------------------------------------------

class TestDeskPlanChoosePlan:
    def test_trial_choose_plan_is_false(self, client):
        tenant = _make_tenant("cp1")
        _make_user(tenant)
        _make_subscription(tenant.id, plan_code=PLAN_TRIAL, status=STATUS_TRIAL)
        token = _login(client, "op@example.com")
        data = client.get(
            "/desk/plan", headers={"Authorization": f"Bearer {token}"}
        ).json()
        assert data["choose_plan"] is False

    def test_active_choose_plan_is_false(self, client):
        tenant = _make_tenant("cp2")
        _make_user(tenant)
        _make_subscription(tenant.id, plan_code=PLAN_DESK, status=STATUS_ACTIVE,
                            trial_end=None, token_cap=TOKEN_CAP[PLAN_DESK])
        token = _login(client, "op@example.com")
        data = client.get(
            "/desk/plan", headers={"Authorization": f"Bearer {token}"}
        ).json()
        assert data["choose_plan"] is False

    def test_expired_choose_plan_is_true(self, client):
        tenant = _make_tenant("cp3")
        _make_user(tenant)
        _make_subscription(
            tenant.id, plan_code=PLAN_TRIAL, status=STATUS_TRIAL,
            trial_end=datetime.now(timezone.utc) - timedelta(days=1),
        )
        token = _login(client, "op@example.com")
        data = client.get(
            "/desk/plan", headers={"Authorization": f"Bearer {token}"}
        ).json()
        # refresh_status advances to expired; choose_plan must be True
        assert data["choose_plan"] is True
        assert data["status"] == STATUS_EXPIRED

    def test_cancelled_choose_plan_is_true(self, client):
        tenant = _make_tenant("cp4")
        _make_user(tenant)
        _make_subscription(tenant.id, plan_code=PLAN_DESK, status=STATUS_CANCELLED,
                            trial_end=None, token_cap=TOKEN_CAP[PLAN_DESK])
        token = _login(client, "op@example.com")
        data = client.get(
            "/desk/plan", headers={"Authorization": f"Bearer {token}"}
        ).json()
        assert data["choose_plan"] is True

    def test_no_subscription_choose_plan_is_true(self, client):
        """A tenant without a subscription row gets choose_plan=True."""
        tenant = _make_tenant("cp5")
        _make_user(tenant)
        # Intentionally no subscription row created.
        token = _login(client, "op@example.com")
        data = client.get(
            "/desk/plan", headers={"Authorization": f"Bearer {token}"}
        ).json()
        assert data["choose_plan"] is True

    def test_trial_days_left_positive_for_live_trial(self, client):
        tenant = _make_tenant("cp6")
        _make_user(tenant)
        _make_subscription(tenant.id)  # default: live 10-day trial
        token = _login(client, "op@example.com")
        data = client.get(
            "/desk/plan", headers={"Authorization": f"Bearer {token}"}
        ).json()
        assert data["trial_days_left"] >= 9  # at most a few seconds elapsed

    def test_trial_days_left_zero_for_paid_plan(self, client):
        tenant = _make_tenant("cp7")
        _make_user(tenant)
        _make_subscription(tenant.id, plan_code=PLAN_AGENTS, status=STATUS_ACTIVE,
                            trial_end=None, token_cap=TOKEN_CAP[PLAN_AGENTS])
        token = _login(client, "op@example.com")
        data = client.get(
            "/desk/plan", headers={"Authorization": f"Bearer {token}"}
        ).json()
        assert data["trial_days_left"] == 0


# ---------------------------------------------------------------------------
# Pipeline — check_plan_capability gating
# ---------------------------------------------------------------------------

class TestPipelinePlanGating:
    def test_desk_plan_refuses_agent_amin(self):
        """Desk plan does not include agents; pipeline returns plan_refused."""
        tenant = _make_tenant("pg1")
        db = SessionLocal()
        try:
            _make_subscription(tenant.id, plan_code=PLAN_DESK, status=STATUS_ACTIVE,
                                trial_end=None, token_cap=TOKEN_CAP[PLAN_DESK])
            msg = _msg(subject="invoice please", provider_message_id="pg1-001",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK_INVOICE)
            assert result.outcome == "plan_refused"
            assert "desk_plan_no_agents" in result.note
        finally:
            db.close()

    def test_mail_plan_refuses_model_when_model_passed(self):
        """Mail plan cannot call a model; pipeline returns plan_refused when model is passed."""
        tenant = _make_tenant("pg2")
        db = SessionLocal()
        try:
            _make_subscription(tenant.id, plan_code=PLAN_MAIL, status=STATUS_ACTIVE,
                                trial_end=None, token_cap=TOKEN_CAP[PLAN_MAIL])
            msg = _msg(subject="general inquiry", provider_message_id="pg2-001",
                       tenant_id=tenant.id)
            model = FakeModel(responses=[
                {"department": "support", "action": "draft_reply", "confidence": 0.9}
            ])
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK_HOLD,
                                  triage_model=model)
            assert result.outcome == "plan_refused"
            assert "mail_plan_no_model" in result.note
        finally:
            db.close()

    def test_agents_plan_at_cap_refuses(self):
        """Agents plan at token cap returns plan_refused."""
        tenant = _make_tenant("pg3")
        db = SessionLocal()
        try:
            cap = TOKEN_CAP[PLAN_AGENTS]
            _make_subscription(tenant.id, plan_code=PLAN_AGENTS, status=STATUS_ACTIVE,
                                trial_end=None, token_cap=cap, tokens_used=cap)
            msg = _msg(subject="invoice check", provider_message_id="pg3-001",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK_INVOICE)
            assert result.outcome == "plan_refused"
            assert "agents_plan_token_cap_reached" in result.note
        finally:
            db.close()

    def test_agents_plan_under_cap_allows(self):
        """Agents plan under cap allows the pipeline to proceed."""
        tenant = _make_tenant("pg4")
        db = SessionLocal()
        try:
            cap = TOKEN_CAP[PLAN_AGENTS]
            _make_subscription(tenant.id, plan_code=PLAN_AGENTS, status=STATUS_ACTIVE,
                                trial_end=None, token_cap=cap, tokens_used=0)
            msg = _msg(subject="invoice approved", provider_message_id="pg4-001",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK_INVOICE)
            assert result.outcome != "plan_refused"
        finally:
            db.close()

    def test_trial_under_cap_allows(self):
        """Trial plan within token cap allows the pipeline."""
        tenant = _make_tenant("pg5")
        db = SessionLocal()
        try:
            _make_subscription(tenant.id, plan_code=PLAN_TRIAL, status=STATUS_TRIAL,
                                token_cap=50_000, tokens_used=0)
            msg = _msg(subject="invoice receipt", provider_message_id="pg5-001",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK_INVOICE)
            assert result.outcome != "plan_refused"
        finally:
            db.close()

    def test_expired_status_refuses(self):
        """Expired subscription returns plan_refused for any agent call."""
        tenant = _make_tenant("pg6")
        db = SessionLocal()
        try:
            _make_subscription(
                tenant.id, plan_code=PLAN_TRIAL, status=STATUS_TRIAL,
                trial_end=datetime.now(timezone.utc) - timedelta(days=1),
                token_cap=50_000, tokens_used=0,
            )
            msg = _msg(subject="invoice late", provider_message_id="pg6-001",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK_INVOICE)
            assert result.outcome == "plan_refused"
            assert "subscription_not_active" in result.note
        finally:
            db.close()

    def test_no_subscription_allows(self):
        """No subscription row → pipeline is not gated (tenant pre-dates commercial phase)."""
        tenant = _make_tenant("pg7")
        db = SessionLocal()
        try:
            # No subscription created: pipeline should not refuse.
            msg = _msg(subject="invoice missing sub", provider_message_id="pg7-001",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK_INVOICE)
            assert result.outcome != "plan_refused"
        finally:
            db.close()
