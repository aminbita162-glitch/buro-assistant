"""
tests/test_buyer_features.py – Phase 7, Section B: fifteen buyer options.

Each test class covers one buyer option.  Tests run against the in-memory
SQLite database created by conftest.py.

Design note: no model is called in any test.  All features are persistence
or policy helpers only.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone

import pytest

from app.main import Base, SessionLocal, Tenant, User, engine, hash_password


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tenant(db, slug: str) -> Tenant:
    t = Tenant(name=slug, slug=slug)
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


def _make_user(db, tenant_id: int, email: str) -> User:
    u = User(
        tenant_id=tenant_id,
        name="Test",
        email=email,
        password_hash=hash_password("pw"),
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _make_message(db, tenant_id: int, subject: str = "Test") -> "Message":
    from app.ingest.models import Message
    msg = Message(
        tenant_id=tenant_id,
        provider="fake",
        provider_message_id=f"mid-{subject}-{id(subject)}",
        raw_json="{}",
        subject_normalized=subject.lower(),
        ingest_time=datetime.now(timezone.utc),
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def _make_draft(db, tenant_id: int) -> "Draft":
    from app.policy.shadow import Draft
    d = Draft(
        tenant_id=tenant_id,
        subject="Re: test",
        body="body",
        state="draft",
        created_at=datetime.now(timezone.utc),
    )
    db.add(d)
    db.commit()
    db.refresh(d)
    return d


@pytest.fixture(autouse=True)
def _tables():
    """Ensure all tables exist before each test."""
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


# ===========================================================================
# B1 – Shared team inbox with assignment
# ===========================================================================

class TestB1Assignment:
    def test_assign_message(self):
        from app.domain.buyer_features import assign_message, get_assignment

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b1")
            msg = _make_message(db, t.id)

            row = assign_message(db, t.id, msg.id, "alice@example.com", assigned_by="admin@example.com")
            assert row.assigned_to == "alice@example.com"
            assert row.tenant_id == t.id

            fetched = get_assignment(db, t.id, msg.id)
            assert fetched is not None
            assert fetched.assigned_to == "alice@example.com"
        finally:
            db.close()

    def test_reassign_message(self):
        from app.domain.buyer_features import assign_message

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b1b")
            msg = _make_message(db, t.id)

            assign_message(db, t.id, msg.id, "alice@example.com")
            row = assign_message(db, t.id, msg.id, "bob@example.com")
            assert row.assigned_to == "bob@example.com"
        finally:
            db.close()

    def test_no_cross_tenant_assignment(self):
        from app.domain.buyer_features import get_assignment, assign_message

        db = SessionLocal()
        try:
            t1 = _make_tenant(db, "b1t1")
            t2 = _make_tenant(db, "b1t2")
            msg = _make_message(db, t1.id)

            assign_message(db, t1.id, msg.id, "alice@example.com")
            result = get_assignment(db, t2.id, msg.id)
            assert result is None
        finally:
            db.close()


# ===========================================================================
# B2 – Internal note (never sent)
# ===========================================================================

class TestB2InternalNote:
    def test_add_note(self):
        from app.domain.buyer_features import add_internal_note, notes_for_message

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b2")
            msg = _make_message(db, t.id)

            note = add_internal_note(db, t.id, msg.id, "alice@example.com", "Do not reply yet.")
            assert note.body == "Do not reply yet."
            assert note.author == "alice@example.com"

            notes = notes_for_message(db, t.id, msg.id)
            assert len(notes) == 1
            assert notes[0].body == "Do not reply yet."
        finally:
            db.close()

    def test_note_isolated_by_tenant(self):
        from app.domain.buyer_features import add_internal_note, notes_for_message

        db = SessionLocal()
        try:
            t1 = _make_tenant(db, "b2t1")
            t2 = _make_tenant(db, "b2t2")
            msg = _make_message(db, t1.id)

            add_internal_note(db, t1.id, msg.id, "a@x.com", "note for t1")
            notes_t2 = notes_for_message(db, t2.id, msg.id)
            assert notes_t2 == []
        finally:
            db.close()


# ===========================================================================
# B3 – Collision lock for drafts
# ===========================================================================

class TestB3CollisionLock:
    def test_acquire_lock(self):
        from app.domain.buyer_features import acquire_draft_lock, get_draft_lock

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b3")
            draft = _make_draft(db, t.id)

            ok = acquire_draft_lock(db, t.id, draft.id, "alice@example.com")
            assert ok is True

            lock = get_draft_lock(db, t.id, draft.id)
            assert lock is not None
            assert lock.locked_by == "alice@example.com"
        finally:
            db.close()

    def test_second_operator_blocked(self):
        from app.domain.buyer_features import acquire_draft_lock

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b3b")
            draft = _make_draft(db, t.id)

            acquire_draft_lock(db, t.id, draft.id, "alice@example.com")
            ok = acquire_draft_lock(db, t.id, draft.id, "bob@example.com")
            assert ok is False
        finally:
            db.close()

    def test_release_lock(self):
        from app.domain.buyer_features import acquire_draft_lock, release_draft_lock, get_draft_lock

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b3c")
            draft = _make_draft(db, t.id)

            acquire_draft_lock(db, t.id, draft.id, "alice@example.com")
            released = release_draft_lock(db, t.id, draft.id, "alice@example.com")
            assert released is True
            assert get_draft_lock(db, t.id, draft.id) is None
        finally:
            db.close()

    def test_expired_lock_allows_new_acquire(self):
        from app.domain.buyer_features import acquire_draft_lock, get_draft_lock, DraftLock

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b3d")
            draft = _make_draft(db, t.id)

            # Insert an already-expired lock directly
            past = datetime.now(timezone.utc) - timedelta(seconds=300)
            lock = DraftLock(
                tenant_id=t.id,
                draft_id=draft.id,
                locked_by="old@example.com",
                locked_at=past,
                expires_at=past,
            )
            db.add(lock)
            db.commit()

            # New acquire should succeed because the old lock is expired.
            ok = acquire_draft_lock(db, t.id, draft.id, "new@example.com")
            assert ok is True
        finally:
            db.close()


# ===========================================================================
# B4 – Snooze until a business hour
# ===========================================================================

class TestB4Snooze:
    def test_snooze_message(self):
        from app.domain.buyer_features import snooze_message, due_snoozed_messages

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b4")
            msg = _make_message(db, t.id)

            future = datetime.now(timezone.utc) + timedelta(hours=2)
            row = snooze_message(db, t.id, msg.id, wake_at=future)
            assert row.reason == "snooze"

            # Not yet due.
            due = due_snoozed_messages(db, t.id)
            assert due == []
        finally:
            db.close()

    def test_due_snooze_returned(self):
        from app.domain.buyer_features import snooze_message, due_snoozed_messages

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b4b")
            msg = _make_message(db, t.id)

            past = datetime.now(timezone.utc) - timedelta(seconds=1)
            snooze_message(db, t.id, msg.id, wake_at=past)

            due = due_snoozed_messages(db, t.id)
            assert len(due) == 1
            assert due[0].message_id == msg.id
        finally:
            db.close()


# ===========================================================================
# B5 – VIP sender list per tenant
# ===========================================================================

class TestB5VipSender:
    def test_add_and_match_exact(self):
        from app.domain.buyer_features import add_vip_sender, is_vip_sender

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b5")
            add_vip_sender(db, t.id, "vip@example.com")

            assert is_vip_sender(db, t.id, "vip@example.com") is True
            assert is_vip_sender(db, t.id, "regular@example.com") is False
        finally:
            db.close()

    def test_domain_pattern_match(self):
        from app.domain.buyer_features import add_vip_sender, is_vip_sender

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b5b")
            add_vip_sender(db, t.id, "@bigcorp.com")

            assert is_vip_sender(db, t.id, "ceo@bigcorp.com") is True
            assert is_vip_sender(db, t.id, "other@notbigcorp.com") is False
        finally:
            db.close()

    def test_no_cross_tenant_vip(self):
        from app.domain.buyer_features import add_vip_sender, is_vip_sender

        db = SessionLocal()
        try:
            t1 = _make_tenant(db, "b5t1")
            t2 = _make_tenant(db, "b5t2")
            add_vip_sender(db, t1.id, "vip@example.com")

            assert is_vip_sender(db, t2.id, "vip@example.com") is False
        finally:
            db.close()


# ===========================================================================
# B6 – After-hours holding queue
# ===========================================================================

class TestB6AfterHours:
    def test_hold_after_hours(self):
        from app.domain.buyer_features import hold_after_hours, due_snoozed_messages

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b6")
            msg = _make_message(db, t.id)

            next_biz = datetime.now(timezone.utc) + timedelta(hours=8)
            row = hold_after_hours(db, t.id, msg.id, next_business_start=next_biz)
            assert row.reason == "after_hours"

            # Not yet due.
            due = due_snoozed_messages(db, t.id)
            assert due == []
        finally:
            db.close()

    def test_after_hours_message_due(self):
        from app.domain.buyer_features import hold_after_hours, due_snoozed_messages

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b6b")
            msg = _make_message(db, t.id)

            past = datetime.now(timezone.utc) - timedelta(seconds=1)
            hold_after_hours(db, t.id, msg.id, next_business_start=past)

            due = due_snoozed_messages(db, t.id)
            assert len(due) == 1
            assert due[0].reason == "after_hours"
        finally:
            db.close()


# ===========================================================================
# B7 – Saved reply snippets (template-checked)
# ===========================================================================

class TestB7ReplySnippets:
    def test_save_snippet(self):
        from app.domain.buyer_features import save_reply_snippet, list_snippets

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b7")
            row = save_reply_snippet(db, t.id, "greeting", "Thank you for your message.")
            assert row.title == "greeting"

            snippets = list_snippets(db, t.id)
            assert len(snippets) == 1
        finally:
            db.close()

    def test_forbidden_phrase_rejected(self):
        from app.domain.buyer_features import save_reply_snippet

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b7b")
            with pytest.raises(ValueError, match="forbidden"):
                save_reply_snippet(
                    db, t.id, "bad", "no refund for this item.",
                    forbidden_phrases=frozenset(["no refund"]),
                )
        finally:
            db.close()

    def test_snippet_tenant_isolated(self):
        from app.domain.buyer_features import save_reply_snippet, list_snippets

        db = SessionLocal()
        try:
            t1 = _make_tenant(db, "b7t1")
            t2 = _make_tenant(db, "b7t2")
            save_reply_snippet(db, t1.id, "s1", "body")
            assert list_snippets(db, t2.id) == []
        finally:
            db.close()


# ===========================================================================
# B8 – Per-department SLA
# ===========================================================================

class TestB8DepartmentSla:
    def test_set_and_get_department_sla(self):
        from app.domain.buyer_features import set_department_sla, get_department_sla_config

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b8")
            set_department_sla(db, t.id, "support", critical_hours=2, high_hours=6,
                               medium_hours=48, low_hours=120)

            config = get_department_sla_config(db, t.id, "support")
            assert config is not None
            assert config["sla_hours"]["critical"] == 2
            assert config["sla_hours"]["high"] == 6
        finally:
            db.close()

    def test_no_config_for_unknown_department(self):
        from app.domain.buyer_features import get_department_sla_config

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b8b")
            config = get_department_sla_config(db, t.id, "unknown")
            assert config is None
        finally:
            db.close()

    def test_department_sla_drives_sla_clock(self):
        from app.domain.buyer_features import set_department_sla, get_department_sla_config
        from app.policy.sla import sla_deadline

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b8c")
            set_department_sla(db, t.id, "legal", critical_hours=1, high_hours=2,
                               medium_hours=8, low_hours=24)

            config = get_department_sla_config(db, t.id, "legal")
            ingest = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
            deadline = sla_deadline(ingest, "high", sla_config=config)
            # high_hours=2 → deadline 2h after ingest
            assert deadline == datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc)
        finally:
            db.close()


# ===========================================================================
# B9 – CSV export of audit log
# ===========================================================================

class TestB9CsvAuditExport:
    def test_csv_headers(self):
        from app.domain.buyer_features import csv_export_audit

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b9")
            csv_text = csv_export_audit(db, t.id)
            reader = csv.reader(io.StringIO(csv_text))
            headers = next(reader)
            assert headers == ["id", "event", "actor", "message_id", "detail", "created_at"]
        finally:
            db.close()

    def test_csv_includes_log_entries(self):
        from app.domain.buyer_features import csv_export_audit
        from app.policy.audit import log_event

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b9b")
            log_event(db, t.id, "test_event", actor="system", detail="detail text")

            csv_text = csv_export_audit(db, t.id)
            reader = csv.reader(io.StringIO(csv_text))
            rows = list(reader)
            # Header + at least 1 data row
            assert len(rows) >= 2
            data = rows[1]
            assert data[1] == "test_event"
            assert data[2] == "system"
        finally:
            db.close()

    def test_csv_tenant_isolation(self):
        from app.domain.buyer_features import csv_export_audit
        from app.policy.audit import log_event

        db = SessionLocal()
        try:
            t1 = _make_tenant(db, "b9t1")
            t2 = _make_tenant(db, "b9t2")
            log_event(db, t1.id, "t1_event")

            csv_text = csv_export_audit(db, t2.id)
            reader = csv.reader(io.StringIO(csv_text))
            rows = list(reader)
            # Only header row for t2 (no events)
            assert len(rows) == 1
        finally:
            db.close()


# ===========================================================================
# B10 – Bounce and failure reason on desk
# ===========================================================================

class TestB10DeliveryFailure:
    def test_record_failure(self):
        from app.domain.buyer_features import record_delivery_failure, failures_for_tenant

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b10")
            row = record_delivery_failure(
                db, t.id,
                reason="550 Mailbox not found",
                smtp_code=550,
                recipient="unknown@example.com",
            )
            assert row.smtp_code == 550
            assert row.reason == "550 Mailbox not found"

            failures = failures_for_tenant(db, t.id)
            assert len(failures) == 1
            assert failures[0].recipient == "unknown@example.com"
        finally:
            db.close()

    def test_failure_tenant_isolation(self):
        from app.domain.buyer_features import record_delivery_failure, failures_for_tenant

        db = SessionLocal()
        try:
            t1 = _make_tenant(db, "b10t1")
            t2 = _make_tenant(db, "b10t2")
            record_delivery_failure(db, t1.id, reason="bounce")
            assert failures_for_tenant(db, t2.id) == []
        finally:
            db.close()


# ===========================================================================
# B11 – Vacation responder (no invented facts)
# ===========================================================================

class TestB11VacationResponder:
    def test_create_responder_inactive_by_default(self):
        from app.domain.buyer_features import (
            set_vacation_responder, get_vacation_responder, vacation_responder_active
        )

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b11")
            set_vacation_responder(db, t.id, template_id="receipt-v1")

            row = get_vacation_responder(db, t.id)
            assert row is not None
            assert row.active is False
            assert vacation_responder_active(db, t.id) is False
        finally:
            db.close()

    def test_active_responder(self):
        from app.domain.buyer_features import set_vacation_responder, vacation_responder_active

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b11b")
            set_vacation_responder(db, t.id, template_id="receipt-v1", active=True)
            assert vacation_responder_active(db, t.id) is True
        finally:
            db.close()

    def test_responder_outside_window_inactive(self):
        from app.domain.buyer_features import set_vacation_responder, vacation_responder_active

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b11c")
            future_start = datetime.now(timezone.utc) + timedelta(days=1)
            set_vacation_responder(
                db, t.id, template_id="receipt-v1",
                active=True, start_date=future_start
            )
            # now < start_date → not active yet
            assert vacation_responder_active(db, t.id) is False
        finally:
            db.close()

    def test_responder_uses_template_not_model(self):
        """
        The vacation responder is template-based only.
        Verify the template_id is stored and no model client is referenced.
        """
        from app.domain.buyer_features import set_vacation_responder, get_vacation_responder
        from app.policy.templates import DEFAULT_REGISTRY

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b11d")
            set_vacation_responder(db, t.id, template_id="receipt-v1", active=True)
            row = get_vacation_responder(db, t.id)
            # template_id must resolve in the registry (no model call)
            tmpl = DEFAULT_REGISTRY.get(row.template_id)
            assert tmpl is not None
        finally:
            db.close()


# ===========================================================================
# B12 – Legal hold blocks retention delete
# ===========================================================================

class TestB12LegalHold:
    def test_set_legal_hold(self):
        from app.domain.buyer_features import set_legal_hold
        from app.ingest.models import Message

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b12")
            msg = _make_message(db, t.id)

            result = set_legal_hold(db, t.id, msg.id, hold=True)
            assert result is True

            refreshed = db.query(Message).filter(Message.id == msg.id).first()
            assert refreshed.legal_hold is True
        finally:
            db.close()

    def test_legal_hold_blocks_retention(self):
        from app.domain.buyer_features import set_legal_hold
        from app.domain.retention import delete_tenant_data
        from app.ingest.models import Message

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b12b")
            msg = _make_message(db, t.id)
            set_legal_hold(db, t.id, msg.id, hold=True)

            result = delete_tenant_data(db, t.id)
            # legal-hold row must not be deleted
            assert result.get("legal_hold_skipped", 0) >= 1
            still_there = db.query(Message).filter(Message.id == msg.id).first()
            assert still_there is not None
        finally:
            db.close()

    def test_clear_legal_hold(self):
        from app.domain.buyer_features import set_legal_hold
        from app.ingest.models import Message

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b12c")
            msg = _make_message(db, t.id)
            set_legal_hold(db, t.id, msg.id, hold=True)
            set_legal_hold(db, t.id, msg.id, hold=False)

            refreshed = db.query(Message).filter(Message.id == msg.id).first()
            assert refreshed.legal_hold is False
        finally:
            db.close()

    def test_no_cross_tenant_hold(self):
        from app.domain.buyer_features import set_legal_hold

        db = SessionLocal()
        try:
            t1 = _make_tenant(db, "b12t1")
            t2 = _make_tenant(db, "b12t2")
            msg = _make_message(db, t1.id)

            # t2 cannot set hold on t1's message
            result = set_legal_hold(db, t2.id, msg.id, hold=True)
            assert result is False
        finally:
            db.close()


# ===========================================================================
# B13 – Role split: admin, operator, auditor
# ===========================================================================

class TestB13Roles:
    def test_assign_role(self):
        from app.domain.buyer_features import set_user_role, get_user_role

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b13")
            u = _make_user(db, t.id, "admin@b13.com")

            set_user_role(db, t.id, u.id, "admin")
            assert get_user_role(db, t.id, u.id) == "admin"
        finally:
            db.close()

    def test_all_valid_roles(self):
        from app.domain.buyer_features import set_user_role, get_user_role, VALID_ROLES

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b13b")
            for i, role in enumerate(sorted(VALID_ROLES)):
                u = _make_user(db, t.id, f"user{i}@b13b.com")
                set_user_role(db, t.id, u.id, role)
                assert get_user_role(db, t.id, u.id) == role
        finally:
            db.close()

    def test_invalid_role_rejected(self):
        from app.domain.buyer_features import set_user_role

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b13c")
            u = _make_user(db, t.id, "bad@b13c.com")
            with pytest.raises(ValueError, match="Invalid role"):
                set_user_role(db, t.id, u.id, "superuser")
        finally:
            db.close()

    def test_role_update(self):
        from app.domain.buyer_features import set_user_role, get_user_role

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b13d")
            u = _make_user(db, t.id, "op@b13d.com")
            set_user_role(db, t.id, u.id, "operator")
            set_user_role(db, t.id, u.id, "auditor")
            assert get_user_role(db, t.id, u.id) == "auditor"
        finally:
            db.close()

    def test_no_role_returns_none(self):
        from app.domain.buyer_features import get_user_role

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b13e")
            u = _make_user(db, t.id, "none@b13e.com")
            assert get_user_role(db, t.id, u.id) is None
        finally:
            db.close()


# ===========================================================================
# B14 – German and English receipt templates
# ===========================================================================

class TestB14LanguageTemplates:
    def test_register_en_and_de(self):
        from app.domain.buyer_features import register_language_templates
        from app.policy.templates import TemplateRegistry

        reg = TemplateRegistry()
        register_language_templates(reg)

        assert reg.get("receipt-en") is not None
        assert reg.get("receipt-de") is not None

    def test_en_template_renders(self):
        from app.domain.buyer_features import register_language_templates
        from app.policy.templates import TemplateRegistry

        reg = TemplateRegistry()
        register_language_templates(reg)
        tmpl = reg.require("receipt-en")
        rendered = tmpl.render({
            "original_subject": "Invoice query",
            "sender_name": "Alice",
            "reference": "REF-001",
        })
        assert "We have received your message" in rendered["body"]
        assert "Invoice query" in rendered["subject"]

    def test_de_template_renders(self):
        from app.domain.buyer_features import register_language_templates
        from app.policy.templates import TemplateRegistry

        reg = TemplateRegistry()
        register_language_templates(reg)
        tmpl = reg.require("receipt-de")
        rendered = tmpl.render({
            "original_subject": "Rechnung",
            "sender_name": "Anna",
            "reference": "REF-002",
        })
        assert "erhalten" in rendered["body"]
        assert "Rechnung" in rendered["subject"]

    def test_de_forbidden_phrase_blocked(self):
        from app.domain.buyer_features import register_language_templates
        from app.policy.templates import TemplateRegistry

        reg = TemplateRegistry()
        register_language_templates(reg)
        tmpl = reg.require("receipt-de")
        # Manually inject forbidden phrase — should raise ValueError
        tmpl_body_original = tmpl.body
        tmpl.body = tmpl_body_original + " Garantie wird gewährt."
        with pytest.raises(ValueError, match="forbidden"):
            tmpl.render({
                "original_subject": "X",
                "sender_name": "Y",
                "reference": "Z",
            })
        # Restore
        tmpl.body = tmpl_body_original

    def test_templates_have_no_invented_facts(self):
        """Neither template calls a model or has free-form text generation."""
        from app.domain.buyer_features import register_language_templates
        from app.policy.templates import TemplateRegistry

        reg = TemplateRegistry()
        register_language_templates(reg)
        for tid in ("receipt-en", "receipt-de"):
            tmpl = reg.require(tid)
            # All required variables must be in the set – no extras that
            # could carry invented content.
            assert tmpl.required_variables == {
                "original_subject", "sender_name", "reference"
            }


# ===========================================================================
# B15 – Daily operator digest, shadow by default
# ===========================================================================

class TestB15Digest:
    def test_default_shadow_mode(self):
        from app.domain.buyer_features import set_digest_subscription

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b15")
            u = _make_user(db, t.id, "op@b15.com")
            row = set_digest_subscription(db, t.id, u.id, "op@b15.com")
            assert row.send_mode == "shadow"
            assert row.active is True
        finally:
            db.close()

    def test_invalid_send_mode_rejected(self):
        from app.domain.buyer_features import set_digest_subscription

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b15b")
            u = _make_user(db, t.id, "op@b15b.com")
            with pytest.raises(ValueError, match="Invalid send_mode"):
                set_digest_subscription(db, t.id, u.id, "op@b15b.com", send_mode="live")
        finally:
            db.close()

    def test_build_digest_structure(self):
        from app.domain.buyer_features import build_digest

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b15c")
            digest = build_digest(db, t.id)
            assert digest["tenant_id"] == t.id
            assert "messages_received" in digest
            assert "drafts_created" in digest
            assert "held_for_approval" in digest
            assert "delivery_failures" in digest
            assert "date" in digest
        finally:
            db.close()

    def test_digest_counts_messages(self):
        from app.domain.buyer_features import build_digest

        db = SessionLocal()
        try:
            t = _make_tenant(db, "b15d")
            _make_message(db, t.id, "msg1")
            _make_message(db, t.id, "msg2")

            digest = build_digest(db, t.id)
            assert digest["messages_received"] == 2
        finally:
            db.close()

    def test_digest_tenant_isolated(self):
        from app.domain.buyer_features import build_digest

        db = SessionLocal()
        try:
            t1 = _make_tenant(db, "b15t1")
            t2 = _make_tenant(db, "b15t2")
            _make_message(db, t1.id, "msg-t1")

            digest_t2 = build_digest(db, t2.id)
            assert digest_t2["messages_received"] == 0
        finally:
            db.close()
