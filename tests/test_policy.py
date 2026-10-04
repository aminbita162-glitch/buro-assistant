"""
Phase 6 policy tests.

Covered checklist rows:
  16 – template registry with variables and forbidden phrases
  17 – auto-reply off unless tenant turns it on
  18 – receipt template states message received and will be reviewed
  19 – business-hours calendar per tenant
  20 – SLA clock from ingest time
  21 – human approval queue
  22 – append-only audit log
  48 – shadow mode: stores draft, does not send

All tests use the in-memory SQLite engine from conftest.py.
No live mail server or model API is contacted.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy import inspect as sa_inspect

from app.main import Base, engine, SessionLocal, Tenant, limiter

# Policy modules
from app.policy.templates import (
    TemplateEntry,
    TemplateRegistry,
    DEFAULT_REGISTRY,
    RECEIPT_TEMPLATE_ID,
    build_receipt_variables,
    check_forbidden_phrases,
)
from app.policy.send_decision import (
    should_send,
    is_send_allowed,
    SEND_DECISION_ALLOW,
    SEND_DECISION_DRAFT,
    SEND_DECISION_SHADOW,
)
from app.policy.calendar import is_business_hours, next_business_start
from app.policy.sla import sla_deadline, is_sla_breached, sla_remaining_seconds
from app.policy.approval import (
    ApprovalQueueEntry,
    enqueue,
    resolve,
    pending_for_tenant,
)
from app.policy.audit import AuditLogEntry, log_event, recent_for_tenant
from app.policy.shadow import Draft, store_draft, is_shadow_mode


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


# ---------------------------------------------------------------------------
# Row 16 – template registry with variables and forbidden phrases
# ---------------------------------------------------------------------------

class TestTemplateRegistry:
    def test_register_and_get(self):
        reg = TemplateRegistry()
        entry = TemplateEntry(
            id="test-tpl",
            subject="Hello {name}",
            body="Dear {name}, your request is being reviewed.",
            required_variables={"name"},
        )
        reg.register(entry)
        assert reg.get("test-tpl") is entry

    def test_require_raises_on_missing(self):
        reg = TemplateRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.require("nonexistent")

    def test_render_substitutes_variables(self):
        reg = TemplateRegistry()
        entry = TemplateEntry(
            id="greet",
            subject="Hi {name}",
            body="Hello {name}.",
            required_variables={"name"},
        )
        reg.register(entry)
        rendered = entry.render({"name": "Alice"})
        assert rendered["body"] == "Hello Alice."
        assert rendered["subject"] == "Hi Alice"

    def test_render_raises_on_missing_variable(self):
        entry = TemplateEntry(
            id="t",
            subject="Hi {name}",
            body="Dear {name}.",
            required_variables={"name"},
        )
        with pytest.raises(KeyError):
            entry.render({})

    def test_forbidden_phrase_rejected(self):
        entry = TemplateEntry(
            id="t2",
            subject="Notice",
            body="We guarantee your satisfaction.",
            required_variables=set(),
            forbidden_phrases=frozenset(["guarantee"]),
        )
        with pytest.raises(ValueError, match="forbidden"):
            entry.render({})

    def test_check_forbidden_phrases_clean(self):
        result = check_forbidden_phrases("Thank you for your message.", frozenset(["guarantee"]))
        assert result == []

    def test_check_forbidden_phrases_hit(self):
        result = check_forbidden_phrases("We guarantee delivery.", frozenset(["guarantee"]))
        assert "guarantee" in result

    def test_list_ids(self):
        reg = TemplateRegistry()
        for i in range(3):
            reg.register(TemplateEntry(id=f"t{i}", subject="", body=""))
        assert len(reg.list_ids()) == 3

    def test_default_registry_has_receipt(self):
        assert DEFAULT_REGISTRY.get(RECEIPT_TEMPLATE_ID) is not None


# ---------------------------------------------------------------------------
# Row 18 – receipt template content
# ---------------------------------------------------------------------------

class TestReceiptTemplate:
    def test_receipt_template_rendered(self):
        entry = DEFAULT_REGISTRY.require(RECEIPT_TEMPLATE_ID)
        variables = build_receipt_variables(
            original_subject="Invoice #123",
            sender_name="Customer",
            reference="REF-001",
        )
        rendered = entry.render(variables)
        assert "received" in rendered["body"].lower()
        assert "reviewed" in rendered["body"].lower()

    def test_receipt_subject_contains_original(self):
        entry = DEFAULT_REGISTRY.require(RECEIPT_TEMPLATE_ID)
        variables = build_receipt_variables("My urgent issue", "Bob", "R-42")
        rendered = entry.render(variables)
        assert "My urgent issue" in rendered["subject"]

    def test_receipt_requires_three_variables(self):
        entry = DEFAULT_REGISTRY.require(RECEIPT_TEMPLATE_ID)
        assert entry.required_variables == {"original_subject", "sender_name", "reference"}

    def test_receipt_body_contains_received_and_reviewed(self):
        entry = DEFAULT_REGISTRY.require(RECEIPT_TEMPLATE_ID)
        body = entry.body
        assert "received" in body.lower()
        assert "reviewed" in body.lower()


# ---------------------------------------------------------------------------
# Row 17 – auto-reply off by default
# ---------------------------------------------------------------------------

class TestSendDecision:
    def test_no_config_returns_draft(self):
        assert should_send(None) == SEND_DECISION_DRAFT

    def test_empty_config_returns_draft(self):
        assert should_send({}) == SEND_DECISION_DRAFT

    def test_auto_reply_enabled_returns_allow(self):
        assert should_send({"auto_reply_enabled": True}) == SEND_DECISION_ALLOW

    def test_auto_reply_false_returns_draft(self):
        assert should_send({"auto_reply_enabled": False}) == SEND_DECISION_DRAFT

    def test_shadow_mode_returns_shadow(self):
        assert should_send({"shadow_mode": True}) == SEND_DECISION_SHADOW

    def test_shadow_mode_overrides_auto_reply(self):
        # Shadow takes priority over auto_reply.
        assert should_send({"shadow_mode": True, "auto_reply_enabled": True}) == SEND_DECISION_SHADOW

    def test_is_send_allowed_only_when_auto_reply(self):
        assert is_send_allowed({"auto_reply_enabled": True}) is True
        assert is_send_allowed(None) is False
        assert is_send_allowed({"shadow_mode": True}) is False

    def test_constants_distinct(self):
        assert SEND_DECISION_ALLOW != SEND_DECISION_DRAFT
        assert SEND_DECISION_ALLOW != SEND_DECISION_SHADOW
        assert SEND_DECISION_DRAFT != SEND_DECISION_SHADOW


# ---------------------------------------------------------------------------
# Row 19 – business-hours calendar
# ---------------------------------------------------------------------------

class TestBusinessHoursCalendar:
    def _dt(self, weekday: int, hour: int) -> datetime:
        """Build a Monday=0 aware UTC datetime with given weekday/hour."""
        # Find a Monday in 2026 and offset by weekday.
        base = datetime(2026, 10, 5, hour, 0, 0, tzinfo=timezone.utc)  # Monday
        return base + timedelta(days=weekday)

    def test_weekday_inside_hours_is_business(self):
        assert is_business_hours(self._dt(0, 10)) is True   # Mon 10:00

    def test_weekend_is_not_business(self):
        assert is_business_hours(self._dt(5, 10)) is False  # Sat 10:00

    def test_before_start_is_not_business(self):
        assert is_business_hours(self._dt(0, 7)) is False   # Mon 07:00

    def test_after_end_is_not_business(self):
        assert is_business_hours(self._dt(0, 17)) is False  # Mon 17:00 (exclusive)

    def test_at_start_hour_is_business(self):
        assert is_business_hours(self._dt(0, 9)) is True    # Mon 09:00

    def test_custom_calendar(self):
        cal = {"working_days": [5, 6], "start_hour": 8, "end_hour": 20}
        assert is_business_hours(self._dt(5, 10), cal) is True   # Sat 10
        assert is_business_hours(self._dt(0, 10), cal) is False  # Mon 10

    def test_next_business_start_from_friday_evening(self):
        friday_evening = datetime(2026, 10, 9, 20, 0, tzinfo=timezone.utc)  # Friday
        nbs = next_business_start(friday_evening)
        assert nbs.weekday() == 0   # Monday
        assert nbs.hour == 9

    def test_next_business_start_during_hours_returns_today(self):
        mon_morning = datetime(2026, 10, 5, 8, 0, tzinfo=timezone.utc)  # Mon before 9
        nbs = next_business_start(mon_morning)
        assert nbs.weekday() == 0
        assert nbs.hour == 9


# ---------------------------------------------------------------------------
# Row 20 – SLA clock from ingest time
# ---------------------------------------------------------------------------

class TestSLAClock:
    def test_critical_deadline_is_1_hour(self):
        t = datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)
        deadline = sla_deadline(t, "critical")
        assert deadline == t + timedelta(hours=1)

    def test_high_deadline_is_4_hours(self):
        t = datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)
        deadline = sla_deadline(t, "high")
        assert deadline == t + timedelta(hours=4)

    def test_medium_deadline_is_24_hours(self):
        t = datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)
        deadline = sla_deadline(t, "medium")
        assert deadline == t + timedelta(hours=24)

    def test_low_deadline_is_72_hours(self):
        t = datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)
        deadline = sla_deadline(t, "low")
        assert deadline == t + timedelta(hours=72)

    def test_custom_sla_config(self):
        t = datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)
        deadline = sla_deadline(t, "critical", {"sla_hours": {"critical": 2}})
        assert deadline == t + timedelta(hours=2)

    def test_not_breached_when_before_deadline(self):
        t = datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)
        now = t + timedelta(minutes=30)
        assert is_sla_breached(t, "critical", now=now) is False

    def test_breached_when_after_deadline(self):
        t = datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)
        now = t + timedelta(hours=2)
        assert is_sla_breached(t, "critical", now=now) is True

    def test_remaining_seconds_positive_before_deadline(self):
        t = datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)
        now = t + timedelta(minutes=30)
        remaining = sla_remaining_seconds(t, "critical", now=now)
        assert remaining > 0

    def test_remaining_seconds_negative_after_deadline(self):
        t = datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)
        now = t + timedelta(hours=2)
        remaining = sla_remaining_seconds(t, "critical", now=now)
        assert remaining < 0

    def test_sla_ingest_time_naive_treated_as_utc(self):
        naive = datetime(2026, 10, 5, 10, 0)  # no tzinfo
        deadline = sla_deadline(naive, "high")
        assert deadline.tzinfo is not None


# ---------------------------------------------------------------------------
# Row 21 – human approval queue
# ---------------------------------------------------------------------------

class TestApprovalQueue:
    def test_tables_exist(self):
        insp = sa_inspect(engine)
        assert "approval_queue" in insp.get_table_names()

    def test_enqueue_creates_pending_entry(self):
        tenant = _make_tenant("aq-t1")
        db = SessionLocal()
        try:
            entry = enqueue(db, tenant.id, "Re: invoice", "Dear customer …", "receipt-v1")
            assert entry.id > 0
            assert entry.state == "pending"
            assert entry.tenant_id == tenant.id
        finally:
            db.close()

    def test_pending_for_tenant_returns_entry(self):
        tenant = _make_tenant("aq-t2")
        db = SessionLocal()
        try:
            enqueue(db, tenant.id, "Subj", "Body")
            pending = pending_for_tenant(db, tenant.id)
            assert len(pending) == 1
        finally:
            db.close()

    def test_approve_entry(self):
        tenant = _make_tenant("aq-t3")
        db = SessionLocal()
        try:
            entry = enqueue(db, tenant.id, "Subj", "Body")
            resolved = resolve(db, entry.id, tenant.id, "approved", "operator@co.com")
            assert resolved is not None
            assert resolved.state == "approved"
            assert resolved.resolved_by == "operator@co.com"
        finally:
            db.close()

    def test_reject_entry(self):
        tenant = _make_tenant("aq-t4")
        db = SessionLocal()
        try:
            entry = enqueue(db, tenant.id, "Subj", "Body")
            resolved = resolve(db, entry.id, tenant.id, "rejected")
            assert resolved.state == "rejected"
        finally:
            db.close()

    def test_cross_tenant_resolve_returns_none(self):
        t1 = _make_tenant("aq-ct1")
        t2 = _make_tenant("aq-ct2")
        db = SessionLocal()
        try:
            entry = enqueue(db, t1.id, "Subj", "Body")
            result = resolve(db, entry.id, t2.id, "approved")
            assert result is None   # wrong tenant — not found
        finally:
            db.close()

    def test_invalid_decision_raises(self):
        tenant = _make_tenant("aq-t5")
        db = SessionLocal()
        try:
            entry = enqueue(db, tenant.id, "Subj", "Body")
            with pytest.raises(ValueError, match="Invalid decision"):
                resolve(db, entry.id, tenant.id, "send")
        finally:
            db.close()

    def test_approved_entry_no_longer_pending(self):
        tenant = _make_tenant("aq-t6")
        db = SessionLocal()
        try:
            entry = enqueue(db, tenant.id, "Subj", "Body")
            resolve(db, entry.id, tenant.id, "approved")
            pending = pending_for_tenant(db, tenant.id)
            assert len(pending) == 0
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Row 22 – append-only audit log
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_table_exists(self):
        insp = sa_inspect(engine)
        assert "audit_log" in insp.get_table_names()

    def test_log_event_creates_row(self):
        tenant = _make_tenant("al-t1")
        db = SessionLocal()
        try:
            entry = log_event(db, tenant.id, "message_ingested")
            assert entry.id > 0
            assert entry.event == "message_ingested"
            assert entry.tenant_id == tenant.id
        finally:
            db.close()

    def test_recent_for_tenant_returns_log(self):
        tenant = _make_tenant("al-t2")
        db = SessionLocal()
        try:
            log_event(db, tenant.id, "triage_decided")
            log_event(db, tenant.id, "draft_created")
            rows = recent_for_tenant(db, tenant.id, limit=10)
            assert len(rows) == 2
        finally:
            db.close()

    def test_log_entries_are_ordered_newest_first(self):
        tenant = _make_tenant("al-t3")
        db = SessionLocal()
        try:
            log_event(db, tenant.id, "event_a")
            log_event(db, tenant.id, "event_b")
            rows = recent_for_tenant(db, tenant.id)
            assert rows[0].event == "event_b"
            assert rows[1].event == "event_a"
        finally:
            db.close()

    def test_audit_log_has_no_update_delete_function(self):
        """No update/delete helper should be exported from audit module."""
        from app.policy import audit as audit_module
        assert not hasattr(audit_module, "update_event")
        assert not hasattr(audit_module, "delete_event")
        assert not hasattr(audit_module, "update_log_entry")

    def test_log_event_with_actor_and_detail(self):
        tenant = _make_tenant("al-t4")
        db = SessionLocal()
        try:
            entry = log_event(
                db, tenant.id, "draft_approved",
                actor="admin@co.com", detail='{"draft_id": 5}'
            )
            assert entry.actor == "admin@co.com"
            assert entry.detail is not None
        finally:
            db.close()

    def test_cross_tenant_logs_isolated(self):
        t1 = _make_tenant("al-ct1")
        t2 = _make_tenant("al-ct2")
        db = SessionLocal()
        try:
            log_event(db, t1.id, "event_t1")
            rows = recent_for_tenant(db, t2.id)
            events = [r.event for r in rows]
            assert "event_t1" not in events
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Row 48 – shadow mode
# ---------------------------------------------------------------------------

class TestShadowMode:
    def test_drafts_table_exists(self):
        insp = sa_inspect(engine)
        assert "drafts" in insp.get_table_names()

    def test_store_draft_default_state_is_draft(self):
        tenant = _make_tenant("sh-t1")
        db = SessionLocal()
        try:
            draft = store_draft(db, tenant.id, "Re: test", "Body text.")
            assert draft.state == SEND_DECISION_DRAFT
        finally:
            db.close()

    def test_shadow_mode_stores_state_shadow(self):
        tenant = _make_tenant("sh-t2")
        db = SessionLocal()
        try:
            draft = store_draft(
                db, tenant.id, "Re: shadow", "Body",
                policy_config={"shadow_mode": True},
            )
            assert draft.state == SEND_DECISION_SHADOW
        finally:
            db.close()

    def test_auto_reply_enabled_still_stores_draft(self):
        """store_draft never sends; it only records the state."""
        tenant = _make_tenant("sh-t3")
        db = SessionLocal()
        try:
            # With auto_reply on the worker (not store_draft) would send.
            # store_draft state is "allow" which maps to SEND_DECISION_ALLOW,
            # but shadow.py stores it as "draft" for non-shadow/non-draft decisions.
            draft = store_draft(
                db, tenant.id, "Re: auto", "Body",
                policy_config={"auto_reply_enabled": True},
            )
            assert draft.id > 0  # stored successfully
        finally:
            db.close()

    def test_is_shadow_mode_true_when_configured(self):
        assert is_shadow_mode({"shadow_mode": True}) is True

    def test_is_shadow_mode_false_by_default(self):
        assert is_shadow_mode(None) is False
        assert is_shadow_mode({"auto_reply_enabled": True}) is False

    def test_shadow_draft_persisted_with_metadata(self):
        tenant = _make_tenant("sh-t4")
        db = SessionLocal()
        try:
            draft = store_draft(
                db, tenant.id,
                subject="Re: invoice",
                body="Your message was received.",
                template_id="receipt-v1",
                language="en",
                decision_hash="abc123",
                policy_config={"shadow_mode": True},
            )
            stored = db.query(Draft).filter(Draft.id == draft.id).first()
            assert stored.template_id == "receipt-v1"
            assert stored.decision_hash == "abc123"
            assert stored.language == "en"
            assert stored.state == SEND_DECISION_SHADOW
        finally:
            db.close()

    def test_shadow_draft_not_sent(self):
        """Shadow mode must never set state to 'sent'."""
        tenant = _make_tenant("sh-t5")
        db = SessionLocal()
        try:
            draft = store_draft(
                db, tenant.id, "Re: x", "Body",
                policy_config={"shadow_mode": True},
            )
            assert draft.state != "sent"
        finally:
            db.close()
