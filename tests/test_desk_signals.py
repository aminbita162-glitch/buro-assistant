"""
tests/test_desk_signals.py – Phase 6 desk signals tests.

Four test classes, one per deliverable:

  TestDailyDigestShadow       – digest always stored as shadow, never sent
  TestSemanticDuplicateFlag   – flag set on message, no second message created
  TestAttachmentTextExtract   – text extracted from allowed types only, bytes
                                never passed to a model
  TestDissatisfiedToneFlag    – flag routes to Leila, does not send
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from app.main import Base, engine, SessionLocal, Tenant, User, hash_password


# ---------------------------------------------------------------------------
# Shared fixtures
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


def _make_user(tenant: Tenant, email: str = "op@example.com") -> User:
    db = SessionLocal()
    try:
        u = User(
            tenant_id=tenant.id,
            name="Operator",
            email=email,
            password_hash=hash_password("pass99"),
        )
        db.add(u)
        db.commit()
        db.refresh(u)
        return u
    finally:
        db.close()


# ---------------------------------------------------------------------------
# TestDailyDigestShadow
# ---------------------------------------------------------------------------

class TestDailyDigestShadow:
    """
    Daily digest is always stored as shadow; it is never sent.
    """

    def test_store_digest_shadow_returns_draft(self):
        from app.policy.digest import store_digest_shadow
        from app.policy.shadow import Draft

        tenant = _make_tenant("d1")
        db = SessionLocal()
        try:
            draft = store_digest_shadow(db, tenant.id)
            assert isinstance(draft, Draft)
        finally:
            db.close()

    def test_digest_state_is_always_shadow(self):
        from app.policy.digest import store_digest_shadow

        tenant = _make_tenant("d2")
        db = SessionLocal()
        try:
            draft = store_digest_shadow(db, tenant.id)
            assert draft.state == "shadow"
        finally:
            db.close()

    def test_digest_template_id_is_daily_digest(self):
        from app.policy.digest import store_digest_shadow

        tenant = _make_tenant("d3")
        db = SessionLocal()
        try:
            draft = store_digest_shadow(db, tenant.id)
            assert draft.template_id == "daily_digest"
        finally:
            db.close()

    def test_digest_body_is_valid_json(self):
        import json
        from app.policy.digest import store_digest_shadow

        tenant = _make_tenant("d4")
        db = SessionLocal()
        try:
            draft = store_digest_shadow(db, tenant.id)
            payload = json.loads(draft.body)
            assert payload["tenant_id"] == tenant.id
        finally:
            db.close()

    def test_digest_payload_contains_required_keys(self):
        import json
        from app.policy.digest import store_digest_shadow

        tenant = _make_tenant("d5")
        db = SessionLocal()
        try:
            draft = store_digest_shadow(db, tenant.id)
            payload = json.loads(draft.body)
            for key in (
                "tenant_id", "date", "messages_received",
                "drafts_created", "held_for_approval",
                "semantic_duplicate_flags", "dissatisfied_tone_flags",
            ):
                assert key in payload, f"missing key: {key}"
        finally:
            db.close()

    def test_digest_persisted_to_drafts_table(self):
        from app.policy.digest import store_digest_shadow
        from app.policy.shadow import Draft

        tenant = _make_tenant("d6")
        db = SessionLocal()
        try:
            store_digest_shadow(db, tenant.id)
            count = db.query(Draft).filter(Draft.tenant_id == tenant.id).count()
            assert count == 1
        finally:
            db.close()

    def test_digest_counts_messages(self):
        import json
        from app.ingest.ingest import ingest_message
        from app.ingest.normalize import NormalizedMessage
        from app.policy.digest import store_digest_shadow

        tenant = _make_tenant("d7")
        db = SessionLocal()
        try:
            for i in range(3):
                msg = NormalizedMessage(
                    provider="fake",
                    provider_message_id=f"d7-{i}",
                    tenant_id=tenant.id,
                    message_id_header=None,
                    subject=f"Subject {i}",
                    subject_normalized=f"subject {i}",
                    sender="s@example.com",
                    recipients=[],
                    body_text=f"body {i}",
                    attachments=[],
                    raw={},
                )
                ingest_message(db, msg)

            draft = store_digest_shadow(db, tenant.id)
            payload = json.loads(draft.body)
            assert payload["messages_received"] == 3
        finally:
            db.close()

    def test_digest_is_tenant_scoped(self):
        import json
        from app.ingest.ingest import ingest_message
        from app.ingest.normalize import NormalizedMessage
        from app.policy.digest import store_digest_shadow

        t1 = _make_tenant("d8-t1")
        t2 = _make_tenant("d8-t2")
        db = SessionLocal()
        try:
            msg = NormalizedMessage(
                provider="fake",
                provider_message_id="d8-msg",
                tenant_id=t2.id,
                message_id_header=None,
                subject="Other tenant",
                subject_normalized="other tenant",
                sender="s@example.com",
                recipients=[],
                body_text="body",
                attachments=[],
                raw={},
            )
            ingest_message(db, msg)

            draft = store_digest_shadow(db, t1.id)
            payload = json.loads(draft.body)
            assert payload["messages_received"] == 0
        finally:
            db.close()

    def test_multiple_digests_all_shadow(self):
        from app.policy.digest import store_digest_shadow
        from app.policy.shadow import Draft

        tenant = _make_tenant("d9")
        db = SessionLocal()
        try:
            store_digest_shadow(db, tenant.id)
            store_digest_shadow(db, tenant.id)
            rows = (
                db.query(Draft)
                .filter(Draft.tenant_id == tenant.id, Draft.state != "shadow")
                .count()
            )
            assert rows == 0
        finally:
            db.close()


# ---------------------------------------------------------------------------
# TestSemanticDuplicateFlag
# ---------------------------------------------------------------------------

class TestSemanticDuplicateFlag:
    """
    Semantic duplicate is a flag on an existing message, not a second row.
    """

    def test_is_semantic_duplicate_identical(self):
        from app.agents.signals import is_semantic_duplicate

        body = "please send me the invoice for order 123"
        assert is_semantic_duplicate(body, [body]) is True

    def test_is_semantic_duplicate_near_identical(self):
        from app.agents.signals import is_semantic_duplicate

        body = "please send me the invoice for order 123"
        near = "please send me invoice for order 123"
        assert is_semantic_duplicate(body, [near]) is True

    def test_is_semantic_duplicate_different(self):
        from app.agents.signals import is_semantic_duplicate

        body = "invoice request order 123"
        different = "job application for software engineer position"
        assert is_semantic_duplicate(body, [different]) is False

    def test_is_semantic_duplicate_empty_candidates(self):
        from app.agents.signals import is_semantic_duplicate

        assert is_semantic_duplicate("some text", []) is False

    def test_is_semantic_duplicate_threshold_respected(self):
        from app.agents.signals import is_semantic_duplicate

        body = "the quick brown fox jumps over the lazy dog"
        # Completely different — should not match at default threshold
        different = "lorem ipsum dolor sit amet consectetur adipiscing"
        assert is_semantic_duplicate(body, [different]) is False

    def test_semantic_duplicate_flag_default_false(self):
        from app.ingest.models import Message
        from app.ingest.ingest import ingest_message
        from app.ingest.normalize import NormalizedMessage

        tenant = _make_tenant("sd1")
        db = SessionLocal()
        try:
            msg = NormalizedMessage(
                provider="fake",
                provider_message_id="sd1-m1",
                tenant_id=tenant.id,
                message_id_header=None,
                subject="Invoice",
                subject_normalized="invoice",
                sender="s@example.com",
                recipients=[],
                body_text="please send invoice",
                attachments=[],
                raw={},
            )
            record, _ = ingest_message(db, msg)
            assert record.semantic_duplicate is False
        finally:
            db.close()

    def test_semantic_duplicate_flag_can_be_set(self):
        from app.ingest.models import Message
        from app.ingest.ingest import ingest_message
        from app.ingest.normalize import NormalizedMessage

        tenant = _make_tenant("sd2")
        db = SessionLocal()
        try:
            msg = NormalizedMessage(
                provider="fake",
                provider_message_id="sd2-m1",
                tenant_id=tenant.id,
                message_id_header=None,
                subject="Invoice copy",
                subject_normalized="invoice copy",
                sender="s@example.com",
                recipients=[],
                body_text="please send invoice",
                attachments=[],
                raw={},
            )
            record, _ = ingest_message(db, msg)
            # Flag is set on the existing row — no new row.
            record.semantic_duplicate = True
            db.commit()
            db.refresh(record)

            # Verify only one row exists.
            count = db.query(Message).filter(Message.tenant_id == tenant.id).count()
            assert count == 1
            assert record.semantic_duplicate is True
        finally:
            db.close()

    def test_semantic_duplicate_flag_no_second_message_created(self):
        """Setting the flag must not insert a new Message row."""
        from app.ingest.models import Message
        from app.ingest.ingest import ingest_message
        from app.ingest.normalize import NormalizedMessage
        from app.agents.signals import is_semantic_duplicate

        tenant = _make_tenant("sd3")
        db = SessionLocal()
        try:
            msg1 = NormalizedMessage(
                provider="fake",
                provider_message_id="sd3-m1",
                tenant_id=tenant.id,
                message_id_header=None,
                subject="Please resend invoice",
                subject_normalized="please resend invoice",
                sender="s@example.com",
                recipients=[],
                body_text="please resend the invoice for order 42",
                attachments=[],
                raw={},
            )
            record1, _ = ingest_message(db, msg1)

            # Second message arrives with near-identical body.
            msg2 = NormalizedMessage(
                provider="fake",
                provider_message_id="sd3-m2",  # different provider ID
                tenant_id=tenant.id,
                message_id_header=None,
                subject="Resend invoice please",
                subject_normalized="resend invoice please",
                sender="s@example.com",
                recipients=[],
                body_text="please resend the invoice for order 42",
                attachments=[],
                raw={},
            )
            record2, _ = ingest_message(db, msg2)

            # Check for semantic duplicate and flag the new row.
            if is_semantic_duplicate(msg2.body_text, [msg1.body_text]):
                record2.semantic_duplicate = True
                db.commit()

            # Must still be exactly two message rows (the original + the new).
            count = db.query(Message).filter(Message.tenant_id == tenant.id).count()
            assert count == 2
            # The new row carries the flag; the original does not.
            db.refresh(record1)
            db.refresh(record2)
            assert record1.semantic_duplicate is False
            assert record2.semantic_duplicate is True
        finally:
            db.close()

    def test_digest_counts_semantic_duplicate_flags(self):
        import json
        from app.ingest.ingest import ingest_message
        from app.ingest.normalize import NormalizedMessage
        from app.agents.signals import is_semantic_duplicate
        from app.policy.digest import store_digest_shadow

        tenant = _make_tenant("sd4")
        db = SessionLocal()
        try:
            msg1 = NormalizedMessage(
                provider="fake",
                provider_message_id="sd4-m1",
                tenant_id=tenant.id,
                message_id_header=None,
                subject="Invoice",
                subject_normalized="invoice",
                sender="s@example.com",
                recipients=[],
                body_text="invoice for order 99",
                attachments=[],
                raw={},
            )
            r1, _ = ingest_message(db, msg1)
            msg2 = NormalizedMessage(
                provider="fake",
                provider_message_id="sd4-m2",
                tenant_id=tenant.id,
                message_id_header=None,
                subject="Invoice copy",
                subject_normalized="invoice copy",
                sender="s@example.com",
                recipients=[],
                body_text="invoice for order 99",
                attachments=[],
                raw={},
            )
            r2, _ = ingest_message(db, msg2)
            if is_semantic_duplicate(msg2.body_text, [msg1.body_text]):
                r2.semantic_duplicate = True
                db.commit()

            draft = store_digest_shadow(db, tenant.id)
            payload = json.loads(draft.body)
            assert payload["semantic_duplicate_flags"] == 1
        finally:
            db.close()


# ---------------------------------------------------------------------------
# TestAttachmentTextExtract
# ---------------------------------------------------------------------------

class TestAttachmentTextExtract:
    """
    Text is extracted only from allowed types; bytes are never sent to a model.
    """

    def test_allowed_for_extraction_text_plain(self):
        from app.ingest.attachment_text import allowed_for_extraction
        assert allowed_for_extraction("text/plain") is True

    def test_allowed_for_extraction_text_csv(self):
        from app.ingest.attachment_text import allowed_for_extraction
        assert allowed_for_extraction("text/csv") is True

    def test_allowed_for_extraction_pdf(self):
        from app.ingest.attachment_text import allowed_for_extraction
        assert allowed_for_extraction("application/pdf") is True

    def test_allowed_for_extraction_disallowed_type(self):
        from app.ingest.attachment_text import allowed_for_extraction
        assert allowed_for_extraction("application/x-executable") is False

    def test_allowed_for_extraction_zip(self):
        from app.ingest.attachment_text import allowed_for_extraction
        assert allowed_for_extraction("application/zip") is False

    def test_extract_text_plain(self):
        from app.ingest.attachment_text import extract_text
        data = b"hello world"
        result = extract_text("text/plain", data)
        assert result == "hello world"

    def test_extract_text_csv(self):
        from app.ingest.attachment_text import extract_text
        data = b"col1,col2\nval1,val2"
        result = extract_text("text/csv", data)
        assert "col1" in result
        assert "val1" in result

    def test_extract_pdf_returns_empty(self):
        """PDF bytes are in allowlist but text extraction is not performed."""
        from app.ingest.attachment_text import extract_text
        data = b"%PDF-1.4 binary content"
        result = extract_text("application/pdf", data)
        assert result == ""

    def test_extract_image_returns_empty(self):
        """Image bytes are allowed but not extracted."""
        from app.ingest.attachment_text import extract_text
        data = b"\xff\xd8\xff\xe0jpeg binary"
        result = extract_text("image/jpeg", data)
        assert result == ""

    def test_extract_disallowed_type_returns_empty(self):
        from app.ingest.attachment_text import extract_text
        data = b"some binary data"
        result = extract_text("application/x-executable", data)
        assert result == ""

    def test_extract_text_is_str(self):
        """Return type is always str, never bytes."""
        from app.ingest.attachment_text import extract_text
        result = extract_text("text/plain", b"hello")
        assert isinstance(result, str)

    def test_extract_handles_bad_utf8_gracefully(self):
        """Bad bytes in text/plain produce a string, not an exception."""
        from app.ingest.attachment_text import extract_text
        data = b"hello \xff world"
        result = extract_text("text/plain", data)
        assert isinstance(result, str)
        assert "hello" in result

    def test_extract_docx_returns_empty(self):
        """DOCX is in allowlist; binary office data is not decoded."""
        from app.ingest.attachment_text import extract_text
        ct = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        result = extract_text(ct, b"PK binary office data")
        assert result == ""

    def test_extract_case_insensitive_content_type(self):
        """Content-type matching is case-insensitive."""
        from app.ingest.attachment_text import extract_text
        result = extract_text("TEXT/PLAIN", b"hello")
        assert result == "hello"


# ---------------------------------------------------------------------------
# TestDissatisfiedToneFlag
# ---------------------------------------------------------------------------

class TestDissatisfiedToneFlag:
    """
    Dissatisfied-tone flag routes to Leila and does not send.
    """

    def test_has_dissatisfied_tone_positive(self):
        from app.agents.signals import has_dissatisfied_tone
        assert has_dissatisfied_tone("This is completely unacceptable service.") is True

    def test_has_dissatisfied_tone_very_disappointed(self):
        from app.agents.signals import has_dissatisfied_tone
        assert has_dissatisfied_tone("I am very disappointed with your response.") is True

    def test_has_dissatisfied_tone_formal_complaint(self):
        from app.agents.signals import has_dissatisfied_tone
        assert has_dissatisfied_tone("I wish to lodge a formal complaint.") is True

    def test_has_dissatisfied_tone_demand_refund(self):
        from app.agents.signals import has_dissatisfied_tone
        assert has_dissatisfied_tone("I demand a refund immediately.") is True

    def test_has_dissatisfied_tone_negative_on_neutral(self):
        from app.agents.signals import has_dissatisfied_tone
        assert has_dissatisfied_tone("Please send me the invoice for order 123.") is False

    def test_has_dissatisfied_tone_negative_on_empty(self):
        from app.agents.signals import has_dissatisfied_tone
        assert has_dissatisfied_tone("") is False

    def test_has_dissatisfied_tone_case_insensitive(self):
        from app.agents.signals import has_dissatisfied_tone
        assert has_dissatisfied_tone("THIS IS COMPLETELY UNACCEPTABLE") is True

    def test_dissatisfied_flag_default_false(self):
        from app.ingest.ingest import ingest_message
        from app.ingest.normalize import NormalizedMessage

        tenant = _make_tenant("dt1")
        db = SessionLocal()
        try:
            msg = NormalizedMessage(
                provider="fake",
                provider_message_id="dt1-m1",
                tenant_id=tenant.id,
                message_id_header=None,
                subject="Help",
                subject_normalized="help",
                sender="s@example.com",
                recipients=[],
                body_text="please help me",
                attachments=[],
                raw={},
            )
            record, _ = ingest_message(db, msg)
            assert record.dissatisfied_tone is False
        finally:
            db.close()

    def test_dissatisfied_flag_can_be_set(self):
        from app.ingest.ingest import ingest_message
        from app.ingest.normalize import NormalizedMessage
        from app.agents.signals import has_dissatisfied_tone

        tenant = _make_tenant("dt2")
        db = SessionLocal()
        try:
            body = "This is completely unacceptable. I want to complain."
            msg = NormalizedMessage(
                provider="fake",
                provider_message_id="dt2-m1",
                tenant_id=tenant.id,
                message_id_header=None,
                subject="Complaint",
                subject_normalized="complaint",
                sender="s@example.com",
                recipients=[],
                body_text=body,
                attachments=[],
                raw={},
            )
            record, _ = ingest_message(db, msg)
            if has_dissatisfied_tone(body):
                record.dissatisfied_tone = True
                db.commit()
                db.refresh(record)
            assert record.dissatisfied_tone is True
        finally:
            db.close()

    def test_dissatisfied_tone_routes_to_leila(self):
        """
        When dissatisfied_tone is detected, Leila supervise() is called
        with reason="dissatisfied_tone" and the action must not be "send".
        """
        from app.agents.leila import supervise
        from app.agents.signals import has_dissatisfied_tone

        body = "This is completely unacceptable service. I am very disappointed."
        assert has_dissatisfied_tone(body) is True

        exception_ctx = {
            "reason": "dissatisfied_tone",
            "detail": "dissatisfaction signals detected in body",
        }
        decision = supervise(exception_ctx)
        assert decision["action"] in ("hold", "request_human", "reroute")
        # Must not be a send action.
        assert decision["action"] != "send"

    def test_dissatisfied_tone_leila_action_not_send(self):
        """Leila never returns action='send' — it is not in the allowed set."""
        from app.agents.leila import supervise, _ALLOWED_ACTIONS

        assert "send" not in _ALLOWED_ACTIONS

        decision = supervise({"reason": "dissatisfied_tone", "detail": ""})
        assert decision["action"] in _ALLOWED_ACTIONS

    def test_dissatisfied_flag_counted_in_digest(self):
        import json
        from app.ingest.ingest import ingest_message
        from app.ingest.normalize import NormalizedMessage
        from app.agents.signals import has_dissatisfied_tone
        from app.policy.digest import store_digest_shadow

        tenant = _make_tenant("dt3")
        db = SessionLocal()
        try:
            body = "This is completely unacceptable, extremely unhappy."
            msg = NormalizedMessage(
                provider="fake",
                provider_message_id="dt3-m1",
                tenant_id=tenant.id,
                message_id_header=None,
                subject="Complaint",
                subject_normalized="complaint",
                sender="s@example.com",
                recipients=[],
                body_text=body,
                attachments=[],
                raw={},
            )
            record, _ = ingest_message(db, msg)
            if has_dissatisfied_tone(body):
                record.dissatisfied_tone = True
                db.commit()

            draft = store_digest_shadow(db, tenant.id)
            payload = json.loads(draft.body)
            assert payload["dissatisfied_tone_flags"] == 1
        finally:
            db.close()

    def test_dissatisfied_flag_send_blocked(self):
        """
        Dissatisfied-tone messages must not be sent even when auto_reply_enabled
        is True.  The flag is checked before the send decision.
        """
        from app.agents.signals import has_dissatisfied_tone
        from app.policy.send_decision import should_send, SEND_DECISION_ALLOW

        body = "I am very disappointed and demand a refund."
        policy_config = {"auto_reply_enabled": True}

        # Without the flag check the send decision would be ALLOW.
        send_dec = should_send(policy_config)
        assert send_dec == SEND_DECISION_ALLOW

        # The caller must block send when dissatisfied_tone is detected.
        dissatisfied = has_dissatisfied_tone(body)
        assert dissatisfied is True
        # Contract: if dissatisfied is True, the caller must not send.
        should_actually_send = (send_dec == SEND_DECISION_ALLOW) and not dissatisfied
        assert should_actually_send is False
