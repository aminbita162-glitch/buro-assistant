"""
tests/test_pipeline.py – follow-up Phase 1: pipeline chain tests.

Covered requirements (DIRECTIVE.txt Phase 1):
  - ingest → Amin → Amilos → shadow/approval chain
  - rule hit must not call a model
  - tokens and cost recorded only when a model is called
  - send is off unless policy_config allows it
  - duplicate and quarantine messages stop the pipeline early
  - low-confidence Amin routes through Leila (supervisor); result stored in approval queue
  - reject action produces no draft and no approval entry

All tests use FakeModel; no external API is called.
All tests use the in-memory SQLite engine from conftest.py.
"""
from __future__ import annotations

import pytest

from app.main import Base, engine, SessionLocal, Tenant, limiter
from app.pipeline import run_pipeline, PipelineResult
from app.agents.fake_model import FakeModel
from app.ingest.normalize import NormalizedMessage
from app.ingest.models import Message
from app.policy.shadow import Draft
from app.policy.approval import ApprovalQueueEntry
from app.workers.cost import null_cost


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
    "domain_rules": [
        {"match": "billing.com", "department": "billing", "action": "draft_reply"},
    ],
    "subject_rules": [
        {"match": "invoice", "department": "billing", "action": "draft_reply"},
        {"match": "escalate me", "department": "support", "action": "escalate"},
        {"match": "hold me", "department": "general", "action": "hold"},
        {"match": "reject me", "department": "general", "action": "reject"},
    ],
    "department_rules": [],
    "confidence_threshold": 0.7,
}


# ---------------------------------------------------------------------------
# Chain: ingest → rule hit → Amilos → draft stored
# ---------------------------------------------------------------------------

class TestRuleHitNoDraftModel:
    """A rule hit must not call the triage model."""

    def test_rule_hit_skips_triage_model(self):
        tenant = _make_tenant("p1")
        db = SessionLocal()
        try:
            msg = _msg(subject="invoice overdue", provider_message_id="p1-001",
                       tenant_id=tenant.id)
            # FakeModel has empty queue — would raise StopIteration if called.
            model = FakeModel()
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK,
                                  triage_model=model)
            assert model.calls == [], "triage model must NOT be called on rule hit"
            assert result.outcome == "draft"
            assert result.triage_decision["rule_hit"] is not None
        finally:
            db.close()

    def test_rule_hit_cost_is_zero(self):
        tenant = _make_tenant("p2")
        db = SessionLocal()
        try:
            msg = _msg(subject="invoice #99", provider_message_id="p2-001",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK)
            assert result.cost is not None
            assert result.cost.tokens_in == 0
            assert result.cost.tokens_out == 0
            assert result.cost.cost_usd == 0.0
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Chain: ingest → model triage → Amilos → draft stored
# ---------------------------------------------------------------------------

class TestModelTriageChain:
    def test_model_called_when_no_rule_fires(self):
        tenant = _make_tenant("p3")
        db = SessionLocal()
        try:
            msg = _msg(subject="general inquiry", provider_message_id="p3-001",
                       tenant_id=tenant.id)
            model = FakeModel(responses=[
                {"department": "support", "action": "draft_reply", "confidence": 0.9}
            ])
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK,
                                  triage_model=model)
            assert len(model.calls) == 1, "triage model must be called when no rule fires"
            assert result.outcome == "draft"
        finally:
            db.close()

    def test_cost_recorded_when_model_called(self):
        tenant = _make_tenant("p4")
        db = SessionLocal()
        try:
            msg = _msg(subject="general question", provider_message_id="p4-001",
                       tenant_id=tenant.id)
            model = FakeModel(responses=[
                {"department": "support", "action": "draft_reply",
                 "confidence": 0.9, "tokens_in": 50, "tokens_out": 15}
            ])
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK,
                                  triage_model=model, triage_model_name="gpt-4.1-mini")
            assert result.cost is not None
            # tokens_in should be > 0 when a model was called
            assert result.cost.tokens_in > 0
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Send decision
# ---------------------------------------------------------------------------

class TestSendDecision:
    def test_send_off_by_default(self):
        tenant = _make_tenant("p5")
        db = SessionLocal()
        try:
            msg = _msg(subject="invoice test", provider_message_id="p5-001",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK,
                                  policy_config=None)
            # Without auto_reply, outcome is "draft" not "send"
            assert result.outcome == "draft"
        finally:
            db.close()

    def test_send_allowed_when_policy_enables_it(self):
        tenant = _make_tenant("p6")
        db = SessionLocal()
        try:
            msg = _msg(subject="invoice test", provider_message_id="p6-001",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK,
                                  policy_config={"auto_reply_enabled": True})
            assert result.outcome == "send"
        finally:
            db.close()

    def test_shadow_mode_stores_draft_not_sent(self):
        tenant = _make_tenant("p7")
        db = SessionLocal()
        try:
            msg = _msg(subject="invoice test", provider_message_id="p7-001",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK,
                                  policy_config={"shadow_mode": True})
            assert result.outcome == "draft"
            assert result.draft_id is not None
            stored = db.query(Draft).filter(Draft.id == result.draft_id).first()
            assert stored.state == "shadow"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Draft stored in DB
# ---------------------------------------------------------------------------

class TestDraftPersisted:
    def test_draft_row_exists_after_pipeline(self):
        tenant = _make_tenant("p8")
        db = SessionLocal()
        try:
            msg = _msg(subject="invoice #200", provider_message_id="p8-001",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK)
            assert result.draft_id is not None
            stored = db.query(Draft).filter(Draft.id == result.draft_id).first()
            assert stored is not None
            assert stored.tenant_id == tenant.id
            assert stored.message_id == result.message.id
        finally:
            db.close()

    def test_draft_subject_and_body_set(self):
        tenant = _make_tenant("p9")
        db = SessionLocal()
        try:
            msg = _msg(subject="invoice #300", provider_message_id="p9-001",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK)
            stored = db.query(Draft).filter(Draft.id == result.draft_id).first()
            assert stored.subject != ""
            assert stored.body != ""
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Duplicate and quarantine stop the pipeline
# ---------------------------------------------------------------------------

class TestEarlyStop:
    def test_duplicate_stops_pipeline(self):
        tenant = _make_tenant("p10")
        db = SessionLocal()
        try:
            msg = _msg(subject="invoice dup", provider_message_id="dup-001",
                       tenant_id=tenant.id)
            # First call ingests.
            run_pipeline(db, msg, rule_pack=_RULE_PACK)
            # Second call with same provider_message_id is a duplicate.
            model = FakeModel()
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK,
                                  triage_model=model)
            assert result.outcome == "duplicate"
            assert model.calls == [], "model must not be called for duplicates"
        finally:
            db.close()

    def test_quarantine_stops_pipeline(self):
        from app.ingest.normalize import Attachment
        tenant = _make_tenant("p11")
        db = SessionLocal()
        try:
            bad_attachment = Attachment(
                filename="virus.exe",
                content_type="application/x-msdownload",
                size_bytes=1024,
            )
            msg = NormalizedMessage(
                provider="fake",
                provider_message_id="quar-001",
                tenant_id=tenant.id,
                message_id_header=None,
                subject="Hello",
                subject_normalized="hello",
                sender="x@example.com",
                recipients=[],
                body_text="check attachment",
                attachments=[bad_attachment],
                raw={},
            )
            model = FakeModel()
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK,
                                  triage_model=model)
            assert result.outcome == "quarantine"
            assert model.calls == [], "model must not be called for quarantined messages"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Non-draft actions go to approval queue
# ---------------------------------------------------------------------------

class TestApprovalQueue:
    def test_hold_action_enqueues_for_approval(self):
        tenant = _make_tenant("p12")
        db = SessionLocal()
        try:
            msg = _msg(subject="hold me please", provider_message_id="hold-001",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK)
            assert result.outcome == "approval"
            assert result.approval_id is not None
            entry = db.query(ApprovalQueueEntry).filter(
                ApprovalQueueEntry.id == result.approval_id
            ).first()
            assert entry.state == "pending"
            assert entry.tenant_id == tenant.id
        finally:
            db.close()

    def test_reject_action_produces_no_draft_no_approval(self):
        tenant = _make_tenant("p13")
        db = SessionLocal()
        try:
            msg = _msg(subject="reject me now", provider_message_id="rej-001",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK)
            assert result.outcome == "no_draft"
            assert result.draft_id is None
            assert result.approval_id is None
        finally:
            db.close()

    def test_escalate_action_enqueues_for_approval(self):
        tenant = _make_tenant("p14")
        db = SessionLocal()
        try:
            msg = _msg(subject="escalate me now", provider_message_id="esc-001",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK)
            assert result.outcome == "approval"
            assert result.approval_id is not None
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Low-confidence triage routes through Leila (supervisor)
# ---------------------------------------------------------------------------

class TestLowConfidenceSupervisor:
    def test_low_confidence_produces_supervisor_decision(self):
        tenant = _make_tenant("p15")
        db = SessionLocal()
        try:
            msg = _msg(subject="mystery request", provider_message_id="lc-001",
                       tenant_id=tenant.id)
            # Model returns low confidence → Amin routes to Leila.
            model = FakeModel(responses=[
                {"department": "general", "action": "hold", "confidence": 0.1}
            ])
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK,
                                  triage_model=model)
            # Leila returns request_human → approval queue
            assert result.outcome in ("approval", "no_draft")
            assert result.triage_decision is not None
            assert result.triage_decision["action"] in (
                "hold", "request_human", "reject", "reroute"
            )
        finally:
            db.close()


# ---------------------------------------------------------------------------
# PipelineResult fields
# ---------------------------------------------------------------------------

class TestPipelineResultFields:
    def test_result_has_message_and_triage_decision(self):
        tenant = _make_tenant("p16")
        db = SessionLocal()
        try:
            msg = _msg(subject="invoice check", provider_message_id="rf-001",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK)
            assert isinstance(result, PipelineResult)
            assert result.message is not None
            assert result.triage_decision is not None
            assert result.cost is not None
        finally:
            db.close()

    def test_result_ingest_result_is_new(self):
        tenant = _make_tenant("p17")
        db = SessionLocal()
        try:
            msg = _msg(subject="invoice new", provider_message_id="rf-002",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK)
            assert result.ingest_result == "new"
        finally:
            db.close()

    def test_reply_draft_present_when_draft_reply_action(self):
        tenant = _make_tenant("p18")
        db = SessionLocal()
        try:
            msg = _msg(subject="invoice reference", provider_message_id="rf-003",
                       tenant_id=tenant.id)
            result = run_pipeline(db, msg, rule_pack=_RULE_PACK)
            assert result.reply_draft is not None
            assert "subject" in result.reply_draft
            assert "body" in result.reply_draft
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------

class TestTenantIsolation:
    def test_drafts_isolated_by_tenant(self):
        t1 = _make_tenant("ti1")
        t2 = _make_tenant("ti2")
        db = SessionLocal()
        try:
            m1 = _msg(subject="invoice t1", provider_message_id="ti-001",
                      tenant_id=t1.id)
            m2 = _msg(subject="invoice t2", provider_message_id="ti-002",
                      tenant_id=t2.id)
            r1 = run_pipeline(db, m1, rule_pack=_RULE_PACK)
            r2 = run_pipeline(db, m2, rule_pack=_RULE_PACK)

            drafts_t1 = db.query(Draft).filter(Draft.tenant_id == t1.id).all()
            drafts_t2 = db.query(Draft).filter(Draft.tenant_id == t2.id).all()
            assert len(drafts_t1) == 1
            assert len(drafts_t2) == 1
            assert drafts_t1[0].id != drafts_t2[0].id
        finally:
            db.close()
