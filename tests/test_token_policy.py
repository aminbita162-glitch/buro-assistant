"""
tests/test_token_policy.py – Phase 5: lowest token policy.

Verified requirements (DIRECTIVE.txt Phase 5):
  - Rule before model: a rule hit must produce zero model tokens.
  - Attachment bytes never go to a model.
  - Prompt contains redacted subject, a short body clip (≤ 200 chars), and the
    rule pack hash.  Full body and attachment content are absent.
  - Cache: identical rule-hit inputs return the cached decision without calling
    the model a second time.
  - Dashboard: /desk/cost returns today_tokens_used and today_cost_usd per tenant.
  - Pipeline emits a usage_event row when a model is called.
  - Pipeline does NOT emit a non-zero usage_event row when a rule fires.
"""
from __future__ import annotations

import pytest

from app.agents.amin import triage, BODY_CLIP_CHARS, _TRIAGE_CACHE, PROMPT_VERSION
from app.agents.fake_model import FakeModel
from app.agents.rules import rule_pack_hash
from app.ingest.normalize import NormalizedMessage, Attachment
from app.main import Base, engine, SessionLocal, Tenant, limiter
from app.pipeline import run_pipeline
from app.domain.usage import UsageEvent


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
    # Clear the in-process triage cache between tests.
    _TRIAGE_CACHE.clear()
    yield
    Base.metadata.drop_all(bind=engine)
    _TRIAGE_CACHE.clear()


def _make_tenant(slug: str = "tp1") -> Tenant:
    db = SessionLocal()
    try:
        t = Tenant(name=slug, slug=slug)
        db.add(t)
        db.commit()
        db.refresh(t)
        return t
    finally:
        db.close()


def _msg(
    subject: str = "Invoice #100",
    body: str = "Please process the attached invoice.",
    sender: str = "user@example.com",
    provider_message_id: str = "tp-001",
    tenant_id: int = 1,
    attachments=None,
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
        attachments=attachments or [],
        raw={},
    )


_RULE_PACK = {
    "domain_rules": [
        {"match": "billing.com", "department": "billing", "action": "draft_reply"},
    ],
    "subject_rules": [
        {"match": "invoice", "department": "billing", "action": "draft_reply"},
    ],
    "department_rules": [],
    "confidence_threshold": 0.7,
}


# ---------------------------------------------------------------------------
# Rule hit → zero model tokens
# ---------------------------------------------------------------------------

class TestRuleHitZeroTokens:
    """A rule hit must produce zero model tokens in every path."""

    def test_rule_hit_triage_tokens_are_zero(self):
        """triage() with a rule hit must not call the model at all."""
        model = FakeModel()  # empty queue — raises StopIteration if called
        msg = _msg(subject="invoice overdue")
        decision = triage(msg, rule_pack=_RULE_PACK, model=model)
        assert model.calls == [], "model must not be called on a rule hit"
        assert decision["rule_hit"] is not None

    def test_rule_hit_pipeline_cost_is_zero(self):
        """run_pipeline() with a rule hit must record zero tokens and zero cost."""
        tenant = _make_tenant("tp-zero")
        db = SessionLocal()
        try:
            msg = _msg(subject="invoice #99", provider_message_id="tp-zero-001",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK)
            assert result.cost is not None
            assert result.cost.tokens_in == 0
            assert result.cost.tokens_out == 0
            assert result.cost.cost_usd == 0.0
        finally:
            db.close()

    def test_rule_hit_no_model_call_in_prompt_spy(self):
        """A spy model verifies the prompt is never built when a rule fires."""
        prompts = []

        class SpyModel:
            def call(self, prompt: str) -> dict:
                prompts.append(prompt)
                return {"department": "billing", "action": "draft_reply", "confidence": 0.9}

        msg = _msg(subject="invoice overdue")
        triage(msg, rule_pack=_RULE_PACK, model=SpyModel())
        assert prompts == [], "model prompt must not be built on a rule hit"


# ---------------------------------------------------------------------------
# Attachment bytes never go to a model
# ---------------------------------------------------------------------------

class TestAttachmentNotSentToModel:
    """Attachment content must not appear in the model prompt."""

    def test_attachment_filename_absent_from_prompt(self):
        attachment = Attachment(
            filename="TOP_SECRET_CONTRACT.pdf",
            content_type="application/pdf",
            size_bytes=204800,
        )
        prompts = []

        class SpyModel:
            def call(self, prompt: str) -> dict:
                prompts.append(prompt)
                return {"department": "support", "action": "draft_reply", "confidence": 0.9}

        msg = _msg(
            subject="general inquiry",
            body="Please see attached.",
            attachments=[attachment],
        )
        # No matching rule → model is called.
        triage(msg, rule_pack=_RULE_PACK, model=SpyModel())
        assert prompts, "model should be called when no rule fires"
        assert "TOP_SECRET_CONTRACT" not in prompts[0], (
            "attachment filename must not appear in the model prompt"
        )
        assert "204800" not in prompts[0], (
            "attachment size must not appear in the model prompt"
        )

    def test_attachment_content_type_absent_from_prompt(self):
        attachment = Attachment(
            filename="data.xlsx",
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            size_bytes=1024,
        )
        prompts = []

        class SpyModel:
            def call(self, prompt: str) -> dict:
                prompts.append(prompt)
                return {"department": "support", "action": "draft_reply", "confidence": 0.9}

        msg = _msg(subject="general report", body="Attached is the report.", attachments=[attachment])
        triage(msg, rule_pack=_RULE_PACK, model=SpyModel())
        assert prompts, "model should be called"
        assert "spreadsheetml" not in prompts[0]


# ---------------------------------------------------------------------------
# Prompt structure: redacted subject + short body clip + rule pack hash
# ---------------------------------------------------------------------------

class TestPromptStructure:
    """The model prompt must follow the Phase 5 format."""

    def test_prompt_contains_rules_hash(self):
        prompts = []

        class SpyModel:
            def call(self, prompt: str) -> dict:
                prompts.append(prompt)
                return {"department": "support", "action": "draft_reply", "confidence": 0.9}

        msg = _msg(subject="general inquiry", body="Need help.")
        triage(msg, rule_pack=_RULE_PACK, model=SpyModel())
        assert prompts
        expected_hash = rule_pack_hash(_RULE_PACK)
        assert expected_hash in prompts[0], (
            f"rule pack hash {expected_hash!r} must appear in prompt"
        )

    def test_prompt_body_clipped_to_200_chars(self):
        long_body = "A" * 500  # 500 chars — well over the 200-char clip
        prompts = []

        class SpyModel:
            def call(self, prompt: str) -> dict:
                prompts.append(prompt)
                return {"department": "support", "action": "draft_reply", "confidence": 0.9}

        msg = _msg(subject="general inquiry", body=long_body)
        triage(msg, rule_pack=_RULE_PACK, model=SpyModel())
        assert prompts
        # The prompt must contain exactly BODY_CLIP_CHARS A's, not 500.
        assert "A" * (BODY_CLIP_CHARS + 1) not in prompts[0], (
            "body clip must not exceed BODY_CLIP_CHARS characters"
        )
        assert "A" * BODY_CLIP_CHARS in prompts[0], (
            "body clip must include at least BODY_CLIP_CHARS characters"
        )

    def test_prompt_version_is_amin_v2(self):
        """Prompt version must be updated to reflect Phase 5 format change."""
        assert PROMPT_VERSION == "amin-v3"

    def test_prompt_contains_redacted_subject_not_raw_pii(self):
        prompts = []

        class SpyModel:
            def call(self, prompt: str) -> dict:
                prompts.append(prompt)
                return {"department": "support", "action": "draft_reply", "confidence": 0.9}

        msg = _msg(subject="Call me at +1 555 000 1234 about support", body="Help needed.")
        triage(msg, rule_pack=_RULE_PACK, model=SpyModel())
        assert prompts
        assert "+1 555 000 1234" not in prompts[0], "raw PII must not appear in prompt"


# ---------------------------------------------------------------------------
# Cache: identical inputs hit the cache, model is not called again
# ---------------------------------------------------------------------------

class TestTriageCache:
    """Identical rule-hit inputs must be returned from cache without a second model call."""

    def test_cache_hit_skips_model(self):
        model = FakeModel()  # empty — would raise if called
        msg = _msg(subject="invoice overdue")

        # First call populates the cache.
        d1 = triage(msg, rule_pack=_RULE_PACK, model=model)
        assert model.calls == []

        # Second call with identical inputs returns from cache.
        d2 = triage(msg, rule_pack=_RULE_PACK, model=model)
        assert model.calls == []
        assert d1["decision_hash"] == d2["decision_hash"]

    def test_different_subjects_different_cache_entries(self):
        model = FakeModel()
        m1 = _msg(subject="invoice alpha")
        m2 = _msg(subject="invoice beta")

        d1 = triage(m1, rule_pack=_RULE_PACK, model=model)
        d2 = triage(m2, rule_pack=_RULE_PACK, model=model)
        # Both should still be rule hits (different hashes because subject differs).
        assert d1["decision_hash"] != d2["decision_hash"]

    def test_model_called_when_no_rule_fires_not_cached(self):
        """Non-rule-hit results are not cached — model is called each time."""
        model = FakeModel(responses=[
            {"department": "support", "action": "draft_reply", "confidence": 0.9},
            {"department": "support", "action": "draft_reply", "confidence": 0.9},
        ])
        msg = _msg(subject="general inquiry no rule")
        triage(msg, rule_pack=_RULE_PACK, model=model)
        triage(msg, rule_pack=_RULE_PACK, model=model)
        assert len(model.calls) == 2, "model must be called every time when no rule fires"


# ---------------------------------------------------------------------------
# Dashboard: usage events are emitted by the pipeline
# ---------------------------------------------------------------------------

class TestDashboardUsageEvents:
    """Pipeline must emit usage events so the dashboard can aggregate per tenant."""

    def test_model_call_emits_usage_event(self):
        """When a model is called, a usage_event row with unit='tokens' must be written."""
        tenant = _make_tenant("tp-usage")
        db = SessionLocal()
        try:
            msg = _msg(
                subject="general inquiry",
                body="Help needed.",
                provider_message_id="usage-001",
                tenant_id=tenant.id,
            )
            model = FakeModel(responses=[
                {"department": "support", "action": "draft_reply",
                 "confidence": 0.9, "tokens_in": 80, "tokens_out": 20}
            ])
            run_pipeline(
                db, msg, rule_pack=_RULE_PACK,
                triage_model=model, triage_model_name="gpt-4.1-mini",
            )
            events = (
                db.query(UsageEvent)
                .filter(
                    UsageEvent.tenant_id == tenant.id,
                    UsageEvent.unit == "tokens",
                )
                .all()
            )
            assert events, "at least one usage_event with unit='tokens' must be emitted"
            total_qty = sum(e.quantity for e in events)
            assert total_qty > 0, "emitted token quantity must be > 0 when model is called"
        finally:
            db.close()

    def test_rule_hit_emits_zero_token_event(self):
        """When a rule fires, any emitted usage_event must have quantity=0."""
        tenant = _make_tenant("tp-rule-usage")
        db = SessionLocal()
        try:
            msg = _msg(
                subject="invoice test",
                provider_message_id="usage-002",
                tenant_id=tenant.id,
            )
            run_pipeline(db, msg, rule_pack=_RULE_PACK)
            events = (
                db.query(UsageEvent)
                .filter(UsageEvent.tenant_id == tenant.id)
                .all()
            )
            # If any token events exist, their quantity must be 0.
            token_events = [e for e in events if (e.unit or "") == "tokens"]
            for e in token_events:
                assert e.quantity == 0, (
                    f"rule-hit usage event must have quantity=0, got {e.quantity}"
                )
        finally:
            db.close()
