"""
Phase 5 contract tests (row 47).

Verifies that each agent module produces output that is valid against the
corresponding JSON schema in schemas/.  Uses jsonschema if installed;
falls back to a required-key assertion if not.

Covered checklist row:
  47 – contract tests for the three schemas
"""
from __future__ import annotations

import json
import os
import pytest

from app.agents.fake_model import FakeModel
from app.agents.amin import triage
from app.agents.amilos import draft_reply
from app.agents.leila import supervise
from app.ingest.normalize import NormalizedMessage


# ---------------------------------------------------------------------------
# Schema loader
# ---------------------------------------------------------------------------

def _load_schema(filename: str) -> dict:
    path = os.path.join(
        os.path.dirname(__file__), "..", "schemas", filename
    )
    with open(os.path.normpath(path), encoding="utf-8") as f:
        return json.load(f)


def _validate(instance: dict, schema: dict) -> None:
    """Validate *instance* against *schema* using jsonschema or key check."""
    try:
        import jsonschema  # type: ignore[import]
        jsonschema.validate(instance=instance, schema=schema)
    except ImportError:
        required = set(schema.get("required", []))
        missing = required - instance.keys()
        assert not missing, f"Missing required fields: {missing}"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_RULE_PACK = {
    "domain_rules": [],
    "subject_rules": [
        {"match": "invoice", "department": "billing", "action": "draft_reply"},
    ],
    "department_rules": [],
    "confidence_threshold": 0.7,
}


def _make_msg(subject: str = "invoice payment", sender: str = "u@client.com") -> NormalizedMessage:
    return NormalizedMessage(
        provider="fake",
        provider_message_id="contract-test-001",
        tenant_id=1,
        message_id_header=None,
        subject=subject,
        subject_normalized=NormalizedMessage.normalize_subject(subject),
        sender=sender,
        recipients=["desk@co.com"],
        body_text="Please process the attached invoice.",
        attachments=[],
        raw={},
    )


# ---------------------------------------------------------------------------
# Row 47 – contract tests for the three schemas
# ---------------------------------------------------------------------------

class TestTriageDecisionContract:
    """TriageDecision output must validate against schemas/triage_decision.json."""

    def test_schema_file_is_valid_json(self):
        schema = _load_schema("triage_decision.json")
        assert "$schema" in schema
        assert schema["type"] == "object"

    def test_schema_has_all_required_fields(self):
        schema = _load_schema("triage_decision.json")
        required = set(schema["required"])
        expected = {
            "schema_version", "prompt_version", "department", "language",
            "urgency", "confidence", "rule_hit", "action", "decision_hash",
        }
        assert expected == required

    def test_amin_output_validates_rule_driven(self):
        schema = _load_schema("triage_decision.json")
        msg = _make_msg()
        decision = triage(msg, rule_pack=_RULE_PACK, model=FakeModel())
        _validate(decision, schema)

    def test_amin_output_validates_model_driven(self):
        schema = _load_schema("triage_decision.json")
        model = FakeModel(responses=[
            {"department": "support", "action": "draft_reply", "confidence": 0.9}
        ])
        msg = _make_msg(subject="general inquiry", sender="u@random.org")
        decision = triage(msg, rule_pack=_RULE_PACK, model=model)
        _validate(decision, schema)

    def test_urgency_enum_valid(self):
        schema = _load_schema("triage_decision.json")
        allowed = set(schema["properties"]["urgency"]["enum"])
        msg = _make_msg()
        decision = triage(msg, rule_pack=_RULE_PACK, model=FakeModel())
        assert decision["urgency"] in allowed

    def test_action_enum_valid(self):
        schema = _load_schema("triage_decision.json")
        allowed = set(schema["properties"]["action"]["enum"])
        msg = _make_msg()
        decision = triage(msg, rule_pack=_RULE_PACK, model=FakeModel())
        assert decision["action"] in allowed

    def test_confidence_in_range(self):
        msg = _make_msg()
        decision = triage(msg, rule_pack=_RULE_PACK, model=FakeModel())
        assert 0.0 <= decision["confidence"] <= 1.0


class TestReplyDraftContract:
    """ReplyDraft output must validate against schemas/reply_draft.json."""

    def test_schema_file_is_valid_json(self):
        schema = _load_schema("reply_draft.json")
        assert "$schema" in schema
        assert schema["type"] == "object"

    def test_schema_has_all_required_fields(self):
        schema = _load_schema("reply_draft.json")
        required = set(schema["required"])
        expected = {"schema_version", "prompt_version", "template_id",
                    "subject", "body", "decision_hash"}
        assert expected == required

    def test_amilos_output_validates(self):
        schema = _load_schema("reply_draft.json")
        triage_dec = {
            "department": "billing",
            "language": "en",
            "urgency": "medium",
            "action": "draft_reply",
            "rule_hit": "subject:invoice",
        }
        draft = draft_reply(
            triage_dec,
            template="Dear {name}, your message has been received.",
            template_id="receipt-v1",
            variables={"name": "Customer"},
        )
        _validate(draft, schema)

    def test_schema_version_is_string(self):
        triage_dec = {"department": "general", "language": "en",
                      "action": "draft_reply", "rule_hit": None}
        draft = draft_reply(triage_dec, template="Hello.", template_id="t", variables={})
        assert isinstance(draft["schema_version"], str)

    def test_body_is_string(self):
        triage_dec = {"department": "general", "language": "en",
                      "action": "draft_reply", "rule_hit": None}
        draft = draft_reply(triage_dec, template="Hello {x}.", template_id="t",
                            variables={"x": "world"})
        assert isinstance(draft["body"], str)


class TestSupervisorDecisionContract:
    """SupervisorDecision output must validate against schemas/supervisor_decision.json."""

    def test_schema_file_is_valid_json(self):
        schema = _load_schema("supervisor_decision.json")
        assert "$schema" in schema
        assert schema["type"] == "object"

    def test_schema_has_all_required_fields(self):
        schema = _load_schema("supervisor_decision.json")
        required = set(schema["required"])
        expected = {"schema_version", "prompt_version", "action", "reason", "decision_hash"}
        assert expected == required

    def test_leila_output_validates_hold(self):
        schema = _load_schema("supervisor_decision.json")
        ctx = {"reason": "schema_invalid", "detail": "missing field"}
        decision = supervise(ctx)
        _validate(decision, schema)

    def test_leila_output_validates_request_human(self):
        schema = _load_schema("supervisor_decision.json")
        ctx = {"reason": "low_confidence"}
        decision = supervise(ctx)
        _validate(decision, schema)

    def test_action_enum_valid(self):
        schema = _load_schema("supervisor_decision.json")
        allowed = set(schema["properties"]["action"]["enum"])
        assert allowed == {"hold", "request_human", "reject", "reroute"}
        ctx = {"reason": "quarantine"}
        decision = supervise(ctx)
        assert decision["action"] in allowed

    def test_reroute_department_null_for_non_reroute(self):
        ctx = {"reason": "low_confidence"}
        decision = supervise(ctx)
        assert decision["reroute_department"] is None

    def test_reroute_department_set_for_reroute(self):
        ctx = {"reason": "no_department", "reroute_department": "billing"}
        decision = supervise(ctx)
        assert decision["action"] == "reroute"
        assert decision["reroute_department"] == "billing"
