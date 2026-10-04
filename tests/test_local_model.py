"""
tests/test_local_model.py – Phase 3: local model route.

Covered requirements (DIRECTIVE.txt Phase 3):
  - Tenant flag ``local_model: true`` prevents the cloud model client from
    being called.
  - When the flag is absent or false the cloud client is used as normal.
  - The fake local client (FakeLocalModel) is used in all tests; no external
    API is called and no vendor key is needed.
  - Both flag states are tested.

All tests use the in-memory SQLite engine from conftest.py.
"""
from __future__ import annotations

import pytest

from app.agents.local_model import FakeLocalModel, is_local_model_enabled
from app.agents.fake_model import FakeModel
from app.agents.amin import triage
from app.ingest.normalize import NormalizedMessage
from app.main import Base, engine, SessionLocal, Tenant, limiter
from app.pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(
    subject: str = "Invoice #100",
    body: str = "Please process the attached invoice.",
    sender: str = "user@example.com",
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
        sender=sender,
        recipients=["desk@co.com"],
        body_text=body,
        attachments=[],
        raw={},
    )


_RULE_PACK = {
    "domain_rules": [],
    "subject_rules": [
        {"match": "invoice", "department": "billing", "action": "draft_reply"},
    ],
    "department_rules": [],
    "confidence_threshold": 0.7,
}


# ---------------------------------------------------------------------------
# Unit tests – flag reader
# ---------------------------------------------------------------------------

class TestIsLocalModelEnabled:
    def test_flag_absent_returns_false(self):
        assert is_local_model_enabled(None) is False
        assert is_local_model_enabled({}) is False

    def test_flag_false_returns_false(self):
        assert is_local_model_enabled({"local_model": False}) is False

    def test_flag_true_returns_true(self):
        assert is_local_model_enabled({"local_model": True}) is True

    def test_other_keys_do_not_matter(self):
        cfg = {"shadow_mode": True, "auto_reply_enabled": False, "local_model": True}
        assert is_local_model_enabled(cfg) is True

    def test_flag_false_with_other_keys(self):
        cfg = {"shadow_mode": False, "local_model": False}
        assert is_local_model_enabled(cfg) is False


# ---------------------------------------------------------------------------
# Unit tests – FakeLocalModel client
# ---------------------------------------------------------------------------

class TestFakeLocalModel:
    def test_call_returns_queued_response(self):
        model = FakeLocalModel(responses=[
            {"department": "billing", "action": "draft_reply", "confidence": 0.9}
        ])
        result = model.call("some prompt")
        assert result["department"] == "billing"
        assert result["action"] == "draft_reply"

    def test_call_records_prompt(self):
        model = FakeLocalModel(responses=[
            {"department": "general", "action": "hold", "confidence": 0.9}
        ])
        model.call("hello")
        assert model.calls == ["hello"]

    def test_call_injects_default_confidence(self):
        model = FakeLocalModel(responses=[{"department": "hr", "action": "hold"}])
        result = model.call("p")
        assert "confidence" in result

    def test_empty_queue_raises_stop_iteration(self):
        model = FakeLocalModel()
        with pytest.raises(StopIteration):
            model.call("p")

    def test_none_sentinel_raises_local_model_error(self):
        from app.agents.local_model import LocalModelError
        model = FakeLocalModel(responses=[None])
        with pytest.raises(LocalModelError):
            model.call("p")

    def test_queue_method_appends(self):
        model = FakeLocalModel()
        model.queue({"department": "support", "action": "hold", "confidence": 0.8})
        result = model.call("x")
        assert result["department"] == "support"

    def test_distinct_from_fake_model(self):
        """FakeLocalModel and FakeModel are different types."""
        local = FakeLocalModel()
        cloud = FakeModel()
        assert type(local) is not type(cloud)


# ---------------------------------------------------------------------------
# Pipeline integration tests – local_model flag
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


class TestLocalModelFlagInPipeline:
    """When local_model flag is set, cloud triage_model must not be called."""

    def test_local_model_used_when_flag_set(self):
        """Pipeline calls FakeLocalModel, not the cloud FakeModel, when flag is True."""
        tenant = _make_tenant("loc1")
        db = SessionLocal()
        try:
            cloud_model = FakeModel()          # empty — would raise if called
            local = FakeLocalModel(responses=[
                {"department": "general", "action": "hold", "confidence": 0.9}
            ])
            result = run_pipeline(
                db,
                _msg(subject="general query", sender="u@example.com", tenant_id=tenant.id),
                rule_pack=None,
                policy_config={"local_model": True},
                triage_model=cloud_model,
                local_model=local,
            )
            # Cloud model must not have been called.
            assert cloud_model.calls == [], "cloud model was called despite local_model flag"
            # Local model must have been called.
            assert len(local.calls) == 1, "local model was not called"
            assert result.outcome in ("approval", "no_draft", "draft", "error")
        finally:
            db.close()

    def test_cloud_model_used_when_flag_absent(self):
        """Pipeline calls the cloud FakeModel when local_model flag is absent."""
        tenant = _make_tenant("loc2")
        db = SessionLocal()
        try:
            cloud_model = FakeModel(responses=[
                {"department": "general", "action": "hold", "confidence": 0.9}
            ])
            local = FakeLocalModel()  # empty — must not be called
            result = run_pipeline(
                db,
                _msg(subject="general query", sender="u@example.com",
                     provider_message_id="msg-002", tenant_id=tenant.id),
                rule_pack=None,
                policy_config={},         # flag absent
                triage_model=cloud_model,
                local_model=local,
            )
            # Cloud model must have been called.
            assert len(cloud_model.calls) == 1, "cloud model was not called"
            # Local model must not have been called.
            assert local.calls == [], "local model was called unexpectedly"
        finally:
            db.close()

    def test_rule_hit_calls_neither_model(self):
        """A rule hit produces a decision without calling cloud or local model."""
        tenant = _make_tenant("loc3")
        db = SessionLocal()
        try:
            cloud_model = FakeModel()      # empty queue — would raise if called
            local = FakeLocalModel()       # empty queue — would raise if called
            result = run_pipeline(
                db,
                _msg(subject="invoice payment", sender="u@example.com",
                     provider_message_id="msg-003", tenant_id=tenant.id),
                rule_pack=_RULE_PACK,
                policy_config={"local_model": True},
                triage_model=cloud_model,
                local_model=local,
            )
            assert cloud_model.calls == [], "cloud model called on rule-hit path"
            assert local.calls == [], "local model called on rule-hit path"
            assert result.triage_decision["rule_hit"] is not None
        finally:
            db.close()
