"""
Phase 5 agent tests.

Covered checklist rows:
  7  – tenant rule engine: domain, subject, department rules
  8  – confidence threshold routes to Leila
  10 – language detection
  11 – urgency lexicon with tenant overrides
  13 – redaction before model call
  14 – prompt/schema version stored; decision hash deterministic
  15 – schema validation of Amin, Amilos, Leila output
  46 – 50 golden messages produce non-error decisions

All tests use FakeModel; no real model API is called.
"""
from __future__ import annotations

import pytest

from app.agents.language import detect_language
from app.agents.urgency import classify_urgency, score_urgency
from app.agents.redact import redact, redact_message
from app.agents.rules import evaluate_rules, get_confidence_threshold, RuleHit
from app.agents.decision_hash import compute_decision_hash
from app.agents.fake_model import FakeModel, ModelCallError
from app.agents.amin import triage, TriageError
from app.agents.amilos import draft_reply, ReplyError
from app.agents.leila import supervise
from app.ingest.normalize import NormalizedMessage
from tests.golden import load_golden_messages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(
    subject: str = "Hello",
    body: str = "Some body text.",
    sender: str = "user@example.com",
    provider_message_id: str = "test-001",
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
        recipients=["desk@company.com"],
        body_text=body,
        attachments=[],
        raw={},
    )


_SIMPLE_RULE_PACK = {
    "domain_rules": [
        {"match": "billing.com", "department": "billing", "action": "draft_reply"},
    ],
    "subject_rules": [
        {"match": "invoice", "department": "billing", "action": "draft_reply"},
        {"match": "job application", "department": "hr", "action": "draft_reply"},
    ],
    "department_rules": [
        {"match": "hr", "department": "hr", "action": "hold"},
    ],
    "confidence_threshold": 0.7,
    "urgency_overrides": {"very special term": 200},
}


# ---------------------------------------------------------------------------
# Row 10 – language detection
# ---------------------------------------------------------------------------

class TestLanguageDetection:
    def test_english_body(self):
        assert detect_language("Please find the invoice attached.") == "en"

    def test_empty_returns_en(self):
        assert detect_language("") == "en"
        assert detect_language("   ") == "en"

    def test_arabic_script_detected(self):
        code = detect_language("لطفاً پاسخ دهید")
        assert code in ("fa", "ar", "ur")   # any Arabic-script code acceptable

    def test_french_heuristic(self):
        code = detect_language("Bonjour, merci pour votre message. Nous vous répondrons.")
        assert code in ("fr", "en")   # en acceptable if langdetect not installed

    def test_returns_string(self):
        result = detect_language("Hello world")
        assert isinstance(result, str)
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# Row 11 – urgency lexicon
# ---------------------------------------------------------------------------

class TestUrgencyLexicon:
    def test_urgent_word_gives_critical(self):
        assert classify_urgency("This is URGENT please reply") == "critical"

    def test_low_priority_text_gives_low(self):
        assert classify_urgency("FYI no rush whenever you can") == "low"

    def test_default_medium(self):
        assert classify_urgency("Please respond to this request") == "medium"

    def test_tenant_override_adds_term(self):
        level = classify_urgency("very special term present", {"very special term": 200})
        assert level == "critical"

    def test_tenant_override_suppresses_term(self):
        # Override sets "urgent" to 0 — should no longer score critical.
        level = classify_urgency("This is URGENT", {"urgent": 0})
        assert level in ("low", "medium")

    def test_score_positive_for_urgent(self):
        assert score_urgency("asap emergency fix") > 0

    def test_score_negative_for_fyi(self):
        assert score_urgency("fyi no rush low priority") < 0


# ---------------------------------------------------------------------------
# Row 13 – redaction before model call
# ---------------------------------------------------------------------------

class TestRedaction:
    def test_email_redacted(self):
        result, count = redact("Contact me at user@example.com for details.")
        assert "[EMAIL]" in result
        assert "user@example.com" not in result
        assert count >= 1

    def test_phone_redacted(self):
        result, count = redact("Call us at +1 800 555 0100")
        assert count >= 1
        assert "+1 800 555 0100" not in result

    def test_clean_text_unchanged(self):
        text = "Please review the attached document."
        result, count = redact(text)
        assert result == text
        assert count == 0

    def test_redact_message_both_fields(self):
        rs, rb, total = redact_message(
            subject="Invoice from user@billing.com",
            body="Pay to account GB29NWBK60161331926819",
        )
        assert "[EMAIL]" in rs
        assert "[IBAN]" in rb
        assert total >= 2

    def test_amin_redacts_before_model(self):
        """Verify Amin does not pass raw PII to the model prompt."""
        calls = []

        class SpyModel:
            def call(self, prompt: str) -> dict:
                calls.append(prompt)
                return {"department": "billing", "action": "draft_reply", "confidence": 0.9}

        msg = _msg(body="Reply to admin@secret.com for details", subject="Test")
        triage(msg, model=SpyModel())
        assert calls, "model was not called"
        assert "admin@secret.com" not in calls[0], "PII leaked to model"


# ---------------------------------------------------------------------------
# Row 7 – rule engine
# ---------------------------------------------------------------------------

class TestRuleEngine:
    def test_domain_rule_fires(self):
        hit = evaluate_rules("billing.com", "subject", "", _SIMPLE_RULE_PACK)
        assert isinstance(hit, RuleHit)
        assert hit.department == "billing"
        assert hit.action == "draft_reply"

    def test_subject_rule_fires(self):
        hit = evaluate_rules("other.com", "invoice payment due", "", _SIMPLE_RULE_PACK)
        assert hit is not None
        assert hit.department == "billing"

    def test_no_rule_returns_none(self):
        hit = evaluate_rules("unknown.com", "random subject", "", _SIMPLE_RULE_PACK)
        assert hit is None

    def test_domain_rule_takes_priority_over_subject(self):
        # Both domain and subject match — domain comes first.
        hit = evaluate_rules("billing.com", "invoice due", "", _SIMPLE_RULE_PACK)
        assert hit.rule_name.startswith("domain:")

    def test_empty_rule_pack_returns_none(self):
        assert evaluate_rules("anything.com", "invoice", "", {}) is None
        assert evaluate_rules("anything.com", "invoice", "", None) is None

    def test_confidence_threshold_default(self):
        assert get_confidence_threshold(None) == 0.7
        assert get_confidence_threshold({"confidence_threshold": 0.5}) == 0.5


# ---------------------------------------------------------------------------
# Row 14 – decision hash determinism
# ---------------------------------------------------------------------------

class TestDecisionHash:
    def test_same_inputs_same_hash(self):
        h1 = compute_decision_hash("invoice payment", "subject:invoice", "draft_reply", _SIMPLE_RULE_PACK)
        h2 = compute_decision_hash("invoice payment", "subject:invoice", "draft_reply", _SIMPLE_RULE_PACK)
        assert h1 == h2

    def test_different_action_different_hash(self):
        h1 = compute_decision_hash("hello", None, "draft_reply", None)
        h2 = compute_decision_hash("hello", None, "hold", None)
        assert h1 != h2

    def test_different_subject_different_hash(self):
        h1 = compute_decision_hash("hello", None, "draft_reply", None)
        h2 = compute_decision_hash("goodbye", None, "draft_reply", None)
        assert h1 != h2

    def test_hash_is_hex_string(self):
        h = compute_decision_hash("test", None, "hold", None)
        assert isinstance(h, str)
        assert len(h) == 64
        int(h, 16)   # raises if not hex


# ---------------------------------------------------------------------------
# Row 7, 8, 14, 15 – Amin triage controller
# ---------------------------------------------------------------------------

class TestAmin:
    def test_rule_hit_no_model_call(self):
        model = FakeModel()  # empty queue — would raise if called
        msg = _msg(subject="invoice overdue", sender="u@billing.com")
        decision = triage(msg, rule_pack=_SIMPLE_RULE_PACK, model=model)
        assert decision["action"] == "draft_reply"
        assert decision["department"] == "billing"
        assert decision["rule_hit"] is not None
        assert model.calls == []   # model was NOT called

    def test_no_rule_calls_model(self):
        model = FakeModel(responses=[
            {"department": "support", "action": "draft_reply", "confidence": 0.95}
        ])
        msg = _msg(subject="general inquiry", sender="u@random.org")
        decision = triage(msg, rule_pack=_SIMPLE_RULE_PACK, model=model)
        assert decision["department"] == "support"
        assert len(model.calls) == 1

    def test_low_confidence_routes_to_leila(self):
        model = FakeModel(responses=[
            {"department": "general", "action": "hold", "confidence": 0.2}
        ])
        msg = _msg(subject="unclear message", sender="u@random.org")
        decision = triage(msg, rule_pack=_SIMPLE_RULE_PACK, model=model)
        # Leila returns a supervisor decision
        assert decision["action"] in ("hold", "request_human", "reject", "reroute")
        assert "schema_version" in decision

    def test_no_rule_no_model_raises(self):
        msg = _msg(subject="uncategorised", sender="x@nowhere.com")
        with pytest.raises(TriageError):
            triage(msg, rule_pack=None, model=None)

    def test_decision_has_required_fields_when_rule_hits(self):
        model = FakeModel()
        msg = _msg(subject="invoice", sender="u@other.com")
        decision = triage(msg, rule_pack=_SIMPLE_RULE_PACK, model=model)
        for field in ("schema_version", "prompt_version", "department", "language",
                      "urgency", "confidence", "rule_hit", "action", "decision_hash"):
            assert field in decision, f"missing field: {field}"

    def test_schema_version_and_prompt_version_set(self):
        model = FakeModel()
        msg = _msg(subject="invoice", sender="u@other.com")
        decision = triage(msg, rule_pack=_SIMPLE_RULE_PACK, model=model)
        assert decision["schema_version"] == "1"
        assert decision["prompt_version"] == "amin-v2"

    def test_deterministic_hash_without_model(self):
        """Same inputs → same hash even when model is disabled."""
        model = FakeModel()
        msg = _msg(subject="invoice payment", sender="u@billing.com")
        d1 = triage(msg, rule_pack=_SIMPLE_RULE_PACK, model=model)
        d2 = triage(msg, rule_pack=_SIMPLE_RULE_PACK, model=FakeModel())
        assert d1["decision_hash"] == d2["decision_hash"]

    def test_urgency_in_decision(self):
        model = FakeModel()
        msg = _msg(subject="urgent invoice", body="URGENT payment overdue", sender="u@billing.com")
        decision = triage(msg, rule_pack=_SIMPLE_RULE_PACK, model=model)
        assert decision["urgency"] in ("critical", "high", "medium", "low")

    def test_language_in_decision(self):
        model = FakeModel()
        msg = _msg(subject="invoice", sender="u@billing.com")
        decision = triage(msg, rule_pack=_SIMPLE_RULE_PACK, model=model)
        assert isinstance(decision["language"], str)
        assert len(decision["language"]) >= 2


# ---------------------------------------------------------------------------
# Row 15 – Amilos reply agent
# ---------------------------------------------------------------------------

class TestAmilos:
    def test_draft_reply_no_model(self):
        triage_dec = {
            "department": "billing",
            "language": "en",
            "urgency": "medium",
            "action": "draft_reply",
            "rule_hit": "subject:invoice",
        }
        draft = draft_reply(
            triage_dec,
            template="Dear {name}, we received your message.",
            template_id="receipt-v1",
            variables={"name": "Customer"},
        )
        assert draft["body"] == "Dear Customer, we received your message."
        assert draft["schema_version"] == "1"
        assert draft["template_id"] == "receipt-v1"
        assert "decision_hash" in draft

    def test_draft_reply_model_used(self):
        model = FakeModel(responses=[
            {"subject": "Re: your message", "body": "Thank you for contacting us."}
        ])
        triage_dec = {"department": "support", "language": "en",
                      "action": "draft_reply", "rule_hit": None}
        draft = draft_reply(
            triage_dec,
            template="{body}",
            template_id="support-v1",
            model=model,
            variables={"body": "default"},
        )
        assert draft["body"] == "Thank you for contacting us."
        assert len(model.calls) == 1

    def test_missing_variable_raises(self):
        triage_dec = {"department": "hr", "language": "en",
                      "action": "draft_reply", "rule_hit": None}
        with pytest.raises(ReplyError):
            draft_reply(triage_dec, template="Hello {missing}", template_id="t",
                        variables={})

    def test_forbidden_price_rejected(self):
        triage_dec = {"department": "billing", "language": "en",
                      "action": "draft_reply", "rule_hit": None}
        with pytest.raises(ValueError, match="forbidden"):
            draft_reply(triage_dec, template="Your price is $1,234.56",
                        template_id="bad", variables={})

    def test_required_fields_present(self):
        triage_dec = {"department": "general", "language": "en",
                      "action": "draft_reply", "rule_hit": None}
        draft = draft_reply(triage_dec, template="Message received.",
                            template_id="generic-v1", variables={})
        for f in ("schema_version", "prompt_version", "template_id",
                  "subject", "body", "decision_hash"):
            assert f in draft


# ---------------------------------------------------------------------------
# Row 15 – Leila supervisor agent
# ---------------------------------------------------------------------------

class TestLeila:
    def test_low_confidence_returns_request_human(self):
        ctx = {"reason": "low_confidence", "detail": "0.2 < 0.7"}
        decision = supervise(ctx)
        assert decision["action"] == "request_human"

    def test_schema_invalid_returns_hold(self):
        ctx = {"reason": "schema_invalid"}
        decision = supervise(ctx)
        assert decision["action"] == "hold"

    def test_quarantine_returns_reject(self):
        ctx = {"reason": "quarantine"}
        decision = supervise(ctx)
        assert decision["action"] == "reject"

    def test_no_department_reroutes(self):
        ctx = {"reason": "no_department", "reroute_department": "general"}
        decision = supervise(ctx)
        assert decision["action"] == "reroute"
        assert decision["reroute_department"] == "general"

    def test_required_fields(self):
        ctx = {"reason": "low_confidence"}
        decision = supervise(ctx)
        for f in ("schema_version", "prompt_version", "action", "reason", "decision_hash"):
            assert f in decision

    def test_unknown_reason_defaults_to_hold(self):
        ctx = {"reason": "mystery_error"}
        decision = supervise(ctx)
        assert decision["action"] == "hold"

    def test_leila_cannot_produce_send_action(self):
        """Supervisor actions are restricted; 'send' and 'delete' are forbidden."""
        ctx = {"reason": "low_confidence"}
        decision = supervise(ctx)
        assert decision["action"] not in ("send", "delete")

    def test_decision_hash_deterministic(self):
        ctx = {"reason": "schema_invalid"}
        d1 = supervise(ctx)
        d2 = supervise(ctx)
        assert d1["decision_hash"] == d2["decision_hash"]


# ---------------------------------------------------------------------------
# Row 46 – 50 golden messages
# ---------------------------------------------------------------------------

class TestGoldenMessages:
    """
    Each golden message is run through Amin using a rule pack that maps
    the expected department.  The test verifies:
    - triage() returns without raising
    - the decision contains all required fields
    - urgency is one of the four valid values
    - decision_hash is a 64-char hex string
    """

    _GOLDEN_RULE_PACK = {
        "domain_rules": [],
        "subject_rules": [
            {"match": "invoice",        "department": "billing",  "action": "draft_reply"},
            {"match": "payment",        "department": "billing",  "action": "draft_reply"},
            {"match": "billing",        "department": "billing",  "action": "draft_reply"},
            {"match": "refund",         "department": "billing",  "action": "draft_reply"},
            {"match": "job application","department": "hr",       "action": "draft_reply"},
            {"match": "leave",          "department": "hr",       "action": "draft_reply"},
            {"match": "onboarding",     "department": "hr",       "action": "draft_reply"},
            {"match": "training",       "department": "hr",       "action": "draft_reply"},
            {"match": "salary",         "department": "hr",       "action": "draft_reply"},
            {"match": "welfare",        "department": "hr",       "action": "draft_reply"},
            {"match": "payroll",        "department": "hr",       "action": "draft_reply"},
            {"match": "expense",        "department": "hr",       "action": "draft_reply"},
            {"match": "support",        "department": "support",  "action": "draft_reply"},
            {"match": "server",         "department": "support",  "action": "escalate"},
            {"match": "security",       "department": "support",  "action": "escalate"},
            {"match": "critical bug",   "department": "support",  "action": "escalate"},
            {"match": "data breach",    "department": "support",  "action": "escalate"},
            {"match": "database migration", "department": "support", "action": "escalate"},
            {"match": "spam",           "department": "general",  "action": "reject"},
            {"match": "suspicious",     "department": "support",  "action": "reject"},
        ],
        "department_rules": [],
        "confidence_threshold": 0.7,
    }

    def test_golden_set_has_50_messages(self):
        messages = load_golden_messages()
        assert len(messages) == 50

    def test_all_golden_messages_produce_valid_decisions(self):
        messages = load_golden_messages()
        errors = []
        for gm in messages:
            subject = gm["subject"]
            body = gm["body"]
            sender = gm["sender"]
            msg = NormalizedMessage(
                provider="fake",
                provider_message_id=gm["provider_message_id"],
                tenant_id=1,
                message_id_header=None,
                subject=subject,
                subject_normalized=NormalizedMessage.normalize_subject(subject),
                sender=sender,
                recipients=["desk@company.com"],
                body_text=body,
                attachments=[],
                raw={},
            )
            # Use a FakeModel as fallback if no rule fires.
            fallback_model = FakeModel(responses=[
                {"department": gm["expected_department"],
                 "action": gm["expected_action"],
                 "confidence": 0.85}
            ])
            try:
                decision = triage(msg, rule_pack=self._GOLDEN_RULE_PACK, model=fallback_model)
                required = {"schema_version", "prompt_version", "department", "language",
                            "urgency", "confidence", "rule_hit", "action", "decision_hash"}
                missing = required - decision.keys()
                if missing:
                    errors.append(f"msg {gm['id']}: missing fields {missing}")
                if decision["urgency"] not in ("critical", "high", "medium", "low"):
                    errors.append(f"msg {gm['id']}: bad urgency {decision['urgency']!r}")
                h = decision["decision_hash"]
                if not (isinstance(h, str) and len(h) == 64):
                    errors.append(f"msg {gm['id']}: bad hash {h!r}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"msg {gm['id']}: raised {exc!r}")

        assert not errors, "\n".join(errors)

    def test_golden_decision_hash_deterministic_for_rule_driven(self):
        """Rule-driven decisions must produce the same hash on repeat calls."""
        messages = load_golden_messages()
        rule_msg = next(
            gm for gm in messages if gm["subject"].lower().startswith("invoice")
        )
        subject = rule_msg["subject"]
        msg = NormalizedMessage(
            provider="fake",
            provider_message_id=rule_msg["provider_message_id"],
            tenant_id=1,
            message_id_header=None,
            subject=subject,
            subject_normalized=NormalizedMessage.normalize_subject(subject),
            sender=rule_msg["sender"],
            recipients=[],
            body_text=rule_msg["body"],
            attachments=[],
            raw={},
        )
        d1 = triage(msg, rule_pack=self._GOLDEN_RULE_PACK, model=FakeModel())
        d2 = triage(msg, rule_pack=self._GOLDEN_RULE_PACK, model=FakeModel())
        assert d1["decision_hash"] == d2["decision_hash"]
