"""
Phase 9 commercial controls tests.

Covered checklist rows:
  40 – usage events
  41 – tenant data export job
  42 – retention field, default 180 days
  43 – sandbox tenant seed command
  44 – scoped API keys, stored hashed
  45 – signed outbound webhooks

All tests run against the in-memory SQLite engine from conftest.py.
No external service is contacted.
"""
from __future__ import annotations

import hashlib
import json
import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy import inspect as sa_inspect

from app.main import Base, engine, SessionLocal, Tenant, User, Task, limiter
from app.ingest.models import Message
from app.ingest.normalize import NormalizedMessage
from app.ingest.ingest import ingest_message

# Commercial modules under test
from app.domain.usage import UsageEvent, emit, events_for_tenant
from app.domain.retention import RETENTION_DEFAULT_DAYS, apply_retention
from app.domain.apikeys import (
    ApiKey,
    create_api_key,
    lookup_api_key,
    revoke_api_key,
    list_api_keys,
    KEY_PREFIX,
)
from app.domain.webhooks import (
    WebhookSubscription,
    build_signed_payload,
    verify_signature,
    create_subscription,
    active_subscriptions,
    deactivate_subscription,
)
from app.domain.export import export_tenant
from app.domain.sandbox import SANDBOX_SLUG, SANDBOX_EMAIL


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
        db.add(t); db.commit(); db.refresh(t)
        return t
    finally:
        db.close()


def _make_user(tenant: Tenant, email: str = "u@example.com") -> User:
    db = SessionLocal()
    try:
        u = User(tenant_id=tenant.id, name="U", email=email,
                 password_hash="x")
        db.add(u); db.commit(); db.refresh(u)
        return u
    finally:
        db.close()


def _ingest_msg(tenant_id: int, provider_message_id: str = "m1",
                subject: str = "Test", days_ago: int = 0) -> Message:
    """Insert a message; optionally back-date its ingest_time."""
    db = SessionLocal()
    try:
        msg = NormalizedMessage(
            provider="fake", provider_message_id=provider_message_id,
            tenant_id=tenant_id, message_id_header=None,
            subject=subject,
            subject_normalized=NormalizedMessage.normalize_subject(subject),
            sender="s@x.com", recipients=[], body_text="body",
            attachments=[], raw={},
        )
        record, _ = ingest_message(db, msg)
        if days_ago:
            record.ingest_time = (
                datetime.now(timezone.utc) - timedelta(days=days_ago)
            )
            db.commit(); db.refresh(record)
        return record
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Row 40 – usage events
# ---------------------------------------------------------------------------

class TestUsageEvents:
    def test_table_exists(self):
        insp = sa_inspect(engine)
        assert "usage_events" in insp.get_table_names()

    def test_emit_creates_row(self):
        tenant = _make_tenant("ue1")
        db = SessionLocal()
        try:
            ev = emit(db, tenant.id, "message_ingested")
            assert ev.id > 0
            assert ev.event_type == "message_ingested"
            assert ev.tenant_id == tenant.id
        finally:
            db.close()

    def test_emit_with_quantity_and_unit(self):
        tenant = _make_tenant("ue2")
        db = SessionLocal()
        try:
            ev = emit(db, tenant.id, "model_called",
                      quantity=150, unit="tokens", cost_usd=0.001)
            assert ev.quantity == 150
            assert ev.unit == "tokens"
            assert ev.cost_usd == pytest.approx(0.001)
        finally:
            db.close()

    def test_events_for_tenant_returns_list(self):
        tenant = _make_tenant("ue3")
        db = SessionLocal()
        try:
            emit(db, tenant.id, "message_ingested")
            emit(db, tenant.id, "draft_sent")
            rows = events_for_tenant(db, tenant.id)
            assert len(rows) == 2
        finally:
            db.close()

    def test_events_filtered_by_type(self):
        tenant = _make_tenant("ue4")
        db = SessionLocal()
        try:
            emit(db, tenant.id, "message_ingested")
            emit(db, tenant.id, "model_called")
            rows = events_for_tenant(db, tenant.id, event_type="model_called")
            assert all(r.event_type == "model_called" for r in rows)
        finally:
            db.close()

    def test_events_are_tenant_scoped(self):
        t1 = _make_tenant("ue-t1")
        t2 = _make_tenant("ue-t2")
        db = SessionLocal()
        try:
            emit(db, t1.id, "message_ingested")
            rows = events_for_tenant(db, t2.id)
            assert len(rows) == 0
        finally:
            db.close()

    def test_emit_is_append_only(self):
        """emit() never updates — each call creates a new row."""
        tenant = _make_tenant("ue5")
        db = SessionLocal()
        try:
            emit(db, tenant.id, "api_key_used")
            emit(db, tenant.id, "api_key_used")
            rows = events_for_tenant(db, tenant.id)
            assert len(rows) == 2
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Row 41 – tenant data export job
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_returns_tenant_record(self):
        tenant = _make_tenant("exp1")
        db = SessionLocal()
        try:
            result = export_tenant(db, tenant.id)
            assert result["tenant"]["slug"] == "exp1"
        finally:
            db.close()

    def test_export_includes_users(self):
        tenant = _make_tenant("exp2")
        _make_user(tenant, "exp@example.com")
        db = SessionLocal()
        try:
            result = export_tenant(db, tenant.id)
            emails = [u["email"] for u in result["users"]]
            assert "exp@example.com" in emails
        finally:
            db.close()

    def test_export_includes_messages(self):
        tenant = _make_tenant("exp3")
        _ingest_msg(tenant.id, "exp-m1", "Export test")
        db = SessionLocal()
        try:
            result = export_tenant(db, tenant.id)
            assert len(result["messages"]) >= 1
        finally:
            db.close()

    def test_export_excludes_raw_json_by_default(self):
        tenant = _make_tenant("exp4")
        _ingest_msg(tenant.id, "exp-m2", "Raw JSON test")
        db = SessionLocal()
        try:
            result = export_tenant(db, tenant.id)
            for m in result["messages"]:
                assert "raw_json" not in m
        finally:
            db.close()

    def test_export_includes_raw_json_when_requested(self):
        tenant = _make_tenant("exp5")
        _ingest_msg(tenant.id, "exp-m3", "Raw JSON include")
        db = SessionLocal()
        try:
            result = export_tenant(db, tenant.id, include_raw_json=True)
            assert any("raw_json" in m for m in result["messages"])
        finally:
            db.close()

    def test_export_has_exported_at_timestamp(self):
        tenant = _make_tenant("exp6")
        db = SessionLocal()
        try:
            result = export_tenant(db, tenant.id)
            assert "exported_at" in result
            assert result["exported_at"] is not None
        finally:
            db.close()

    def test_export_unknown_tenant_returns_error(self):
        db = SessionLocal()
        try:
            result = export_tenant(db, 999999)
            assert "error" in result
        finally:
            db.close()

    def test_export_is_json_serialisable(self):
        tenant = _make_tenant("exp7")
        _ingest_msg(tenant.id, "exp-m4", "Serialise test")
        db = SessionLocal()
        try:
            result = export_tenant(db, tenant.id)
            # Must not raise
            json.dumps(result)
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Row 42 – retention field, default 180 days
# ---------------------------------------------------------------------------

class TestRetention:
    def test_retention_default_is_180(self):
        assert RETENTION_DEFAULT_DAYS == 180

    def test_retention_column_in_migration(self):
        """Migration 0006 adds retention_days to tenants."""
        with open("migrations/versions/0006_commercial.py", encoding="utf-8") as f:
            src = f.read()
        assert "retention_days" in src

    def test_apply_retention_deletes_old_messages(self):
        tenant = _make_tenant("ret1")
        _ingest_msg(tenant.id, "old-m1", "Old message", days_ago=200)
        db = SessionLocal()
        try:
            count = apply_retention(db, tenant.id, tenant_retention_days=180)
            assert count >= 1
        finally:
            db.close()

    def test_apply_retention_keeps_recent_messages(self):
        tenant = _make_tenant("ret2")
        _ingest_msg(tenant.id, "new-m1", "New message", days_ago=10)
        db = SessionLocal()
        try:
            count = apply_retention(db, tenant.id, tenant_retention_days=180)
            assert count == 0
        finally:
            db.close()

    def test_apply_retention_respects_custom_window(self):
        tenant = _make_tenant("ret3")
        _ingest_msg(tenant.id, "old-m2", "Old 30 days", days_ago=35)
        db = SessionLocal()
        try:
            count = apply_retention(db, tenant.id, tenant_retention_days=30)
            assert count >= 1
        finally:
            db.close()

    def test_apply_retention_is_tenant_scoped(self):
        t1 = _make_tenant("ret-t1")
        t2 = _make_tenant("ret-t2")
        _ingest_msg(t1.id, "old-t1", "Old t1", days_ago=200)
        db = SessionLocal()
        try:
            count = apply_retention(db, t2.id, tenant_retention_days=180)
            assert count == 0  # t1 message not deleted for t2
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Row 43 – sandbox tenant seed command
# ---------------------------------------------------------------------------

class TestSandboxSeed:
    def test_sandbox_module_importable(self):
        from app.domain import sandbox  # noqa: F401

    def test_sandbox_slug_constant(self):
        assert SANDBOX_SLUG == "sandbox"

    def test_sandbox_email_constant(self):
        assert SANDBOX_EMAIL == "sandbox@example.com"

    def test_seed_sandbox_creates_tenant(self):
        from app.domain.sandbox import seed_sandbox
        seed_sandbox()
        db = SessionLocal()
        try:
            tenant = db.query(Tenant).filter(Tenant.slug == SANDBOX_SLUG).first()
            assert tenant is not None
            assert tenant.name == "Sandbox Tenant"
        finally:
            db.close()

    def test_seed_sandbox_creates_user(self):
        from app.domain.sandbox import seed_sandbox
        seed_sandbox()
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == SANDBOX_EMAIL).first()
            assert user is not None
        finally:
            db.close()

    def test_seed_sandbox_is_idempotent(self):
        from app.domain.sandbox import seed_sandbox
        seed_sandbox()
        seed_sandbox()   # second call must not raise or duplicate
        db = SessionLocal()
        try:
            count = db.query(Tenant).filter(Tenant.slug == SANDBOX_SLUG).count()
            assert count == 1
        finally:
            db.close()

    def test_seed_sandbox_creates_task(self):
        from app.domain.sandbox import seed_sandbox
        seed_sandbox()
        db = SessionLocal()
        try:
            tenant = db.query(Tenant).filter(Tenant.slug == SANDBOX_SLUG).first()
            user = db.query(User).filter(User.email == SANDBOX_EMAIL).first()
            task = (
                db.query(Task)
                .filter(Task.tenant_id == tenant.id, Task.user_id == user.id)
                .first()
            )
            assert task is not None
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Row 44 – scoped API keys, stored hashed
# ---------------------------------------------------------------------------

class TestApiKeys:
    def test_table_exists(self):
        insp = sa_inspect(engine)
        assert "api_keys" in insp.get_table_names()

    def test_create_returns_raw_key(self):
        tenant = _make_tenant("ak1")
        db = SessionLocal()
        try:
            entry, raw = create_api_key(db, tenant.id, "my-key", scope="ingest")
            assert raw.startswith(KEY_PREFIX)
            assert len(raw) > len(KEY_PREFIX)
        finally:
            db.close()

    def test_raw_key_not_stored(self):
        tenant = _make_tenant("ak2")
        db = SessionLocal()
        try:
            entry, raw = create_api_key(db, tenant.id, "k", scope="read")
            stored = db.query(ApiKey).filter(ApiKey.id == entry.id).first()
            assert stored.key_hash != raw        # hash, not plaintext
            assert raw not in stored.key_hash    # definitely not in the hash
        finally:
            db.close()

    def test_lookup_by_raw_key(self):
        tenant = _make_tenant("ak3")
        db = SessionLocal()
        try:
            entry, raw = create_api_key(db, tenant.id, "k2")
            found = lookup_api_key(db, raw)
            assert found is not None
            assert found.id == entry.id
        finally:
            db.close()

    def test_lookup_wrong_key_returns_none(self):
        tenant = _make_tenant("ak4")
        db = SessionLocal()
        try:
            create_api_key(db, tenant.id, "k3")
            assert lookup_api_key(db, "wrong-key") is None
        finally:
            db.close()

    def test_revoke_key(self):
        tenant = _make_tenant("ak5")
        db = SessionLocal()
        try:
            entry, raw = create_api_key(db, tenant.id, "k4")
            assert revoke_api_key(db, entry.id, tenant.id) is True
            assert lookup_api_key(db, raw) is None
        finally:
            db.close()

    def test_revoke_wrong_tenant_fails(self):
        t1 = _make_tenant("ak-t1")
        t2 = _make_tenant("ak-t2")
        db = SessionLocal()
        try:
            entry, _ = create_api_key(db, t1.id, "k5")
            assert revoke_api_key(db, entry.id, t2.id) is False
        finally:
            db.close()

    def test_list_api_keys_excludes_revoked(self):
        tenant = _make_tenant("ak6")
        db = SessionLocal()
        try:
            entry, _ = create_api_key(db, tenant.id, "k6")
            revoke_api_key(db, entry.id, tenant.id)
            keys = list_api_keys(db, tenant.id)
            assert len(keys) == 0
        finally:
            db.close()

    def test_key_has_scope(self):
        tenant = _make_tenant("ak7")
        db = SessionLocal()
        try:
            entry, _ = create_api_key(db, tenant.id, "k7", scope="admin")
            assert entry.scope == "admin"
        finally:
            db.close()

    def test_key_hash_is_sha256(self):
        tenant = _make_tenant("ak8")
        db = SessionLocal()
        try:
            entry, raw = create_api_key(db, tenant.id, "k8")
            expected_hash = hashlib.sha256(raw.encode()).hexdigest()
            assert entry.key_hash == expected_hash
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Row 45 – signed outbound webhooks
# ---------------------------------------------------------------------------

class TestWebhooks:
    def test_table_exists(self):
        insp = sa_inspect(engine)
        assert "webhook_subscriptions" in insp.get_table_names()

    def test_create_subscription_returns_secret(self):
        tenant = _make_tenant("wh1")
        db = SessionLocal()
        try:
            sub, secret = create_subscription(db, tenant.id, "https://example.com/hook")
            assert sub.id > 0
            assert len(secret) >= 32
        finally:
            db.close()

    def test_secret_not_stored_in_plaintext(self):
        tenant = _make_tenant("wh2")
        db = SessionLocal()
        try:
            sub, secret = create_subscription(db, tenant.id, "https://x.com/hook")
            stored = db.query(WebhookSubscription).filter(
                WebhookSubscription.id == sub.id
            ).first()
            assert stored.secret_hash != secret
        finally:
            db.close()

    def test_build_signed_payload(self):
        body, sig = build_signed_payload("message_ingested", {"id": 1}, "mysecret")
        assert sig.startswith("sha256=")
        assert isinstance(body, bytes)

    def test_verify_signature_valid(self):
        body, sig = build_signed_payload("draft_sent", {"id": 2}, "s3cr3t")
        assert verify_signature(body, "s3cr3t", sig) is True

    def test_verify_signature_invalid(self):
        body, sig = build_signed_payload("draft_sent", {"id": 2}, "s3cr3t")
        assert verify_signature(body, "wrong-secret", sig) is False

    def test_verify_signature_tampered_body(self):
        body, sig = build_signed_payload("draft_sent", {"id": 2}, "s3cr3t")
        tampered = body + b"x"
        assert verify_signature(tampered, "s3cr3t", sig) is False

    def test_active_subscriptions_returns_active(self):
        tenant = _make_tenant("wh3")
        db = SessionLocal()
        try:
            create_subscription(db, tenant.id, "https://a.com/h")
            create_subscription(db, tenant.id, "https://b.com/h")
            subs = active_subscriptions(db, tenant.id)
            assert len(subs) == 2
        finally:
            db.close()

    def test_deactivate_subscription(self):
        tenant = _make_tenant("wh4")
        db = SessionLocal()
        try:
            sub, _ = create_subscription(db, tenant.id, "https://c.com/h")
            assert deactivate_subscription(db, sub.id, tenant.id) is True
            subs = active_subscriptions(db, tenant.id)
            assert len(subs) == 0
        finally:
            db.close()

    def test_event_filter(self):
        tenant = _make_tenant("wh5")
        db = SessionLocal()
        try:
            create_subscription(db, tenant.id, "https://d.com/h",
                                events="message_ingested,draft_sent")
            create_subscription(db, tenant.id, "https://e.com/h",
                                events="model_called")
            matching = active_subscriptions(db, tenant.id,
                                            event_type="draft_sent")
            assert len(matching) == 1
            assert "draft_sent" in matching[0].events
        finally:
            db.close()

    def test_empty_events_matches_all(self):
        tenant = _make_tenant("wh6")
        db = SessionLocal()
        try:
            create_subscription(db, tenant.id, "https://f.com/h", events="")
            subs = active_subscriptions(db, tenant.id, event_type="anything")
            assert len(subs) == 1
        finally:
            db.close()

    def test_cross_tenant_subscription_isolation(self):
        t1 = _make_tenant("wh-t1")
        t2 = _make_tenant("wh-t2")
        db = SessionLocal()
        try:
            create_subscription(db, t1.id, "https://g.com/h")
            subs = active_subscriptions(db, t2.id)
            assert len(subs) == 0
        finally:
            db.close()
