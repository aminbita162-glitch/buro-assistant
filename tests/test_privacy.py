"""
tests/test_privacy.py – Phase 4 privacy pack tests.

Covered:
  - GET  /desk/privacy/export  – data portability for one tenant, tenant-scoped
  - DELETE /desk/privacy/data  – right-to-erasure, legal-hold blocks deletion
  - apply_retention respects legal_hold
  - delete_tenant_data counts
  - _SecretFilter scrubs IMAP_PASSWORD from log records
  - Redaction before model call (regression guard)

All tests use the in-memory SQLite engine from conftest.py.
No external service is contacted.
"""
from __future__ import annotations

import logging
import os
import pytest
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import (
    app, Base, engine, SessionLocal,
    Tenant, User, Task,
    hash_password, limiter,
)
from app.ingest.models import Message
from app.ingest.normalize import NormalizedMessage
from app.ingest.ingest import ingest_message
from app.policy.shadow import Draft
from app.policy.approval import ApprovalQueueEntry, enqueue
from app.policy.audit import log_event
from app.domain.retention import apply_retention, delete_tenant_data, RETENTION_DEFAULT_DAYS
from app.agents.redact import redact, redact_message


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


def _ingest(tenant_id: int, provider_message_id: str, subject: str) -> Message:
    db = SessionLocal()
    try:
        msg = NormalizedMessage(
            provider="fake",
            provider_message_id=provider_message_id,
            tenant_id=tenant_id,
            message_id_header=None,
            subject=subject,
            subject_normalized=NormalizedMessage.normalize_subject(subject),
            sender="s@example.com",
            recipients=[],
            body_text="body",
            attachments=[],
            raw={},
        )
        record, _ = ingest_message(db, msg)
        return record
    finally:
        db.close()


def _make_draft(tenant_id: int, state: str = "draft") -> Draft:
    db = SessionLocal()
    try:
        d = Draft(
            tenant_id=tenant_id,
            subject="Re: test",
            body="Draft body.",
            state=state,
            created_at=datetime.now(timezone.utc),
        )
        db.add(d); db.commit(); db.refresh(d)
        return d
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Export endpoint
# ---------------------------------------------------------------------------

class TestPrivacyExport:
    def test_export_requires_auth(self, client):
        r = client.get("/desk/privacy/export")
        assert r.status_code == 401

    def test_export_returns_200(self, client):
        tenant = _make_tenant()
        _make_user(tenant)
        token = _login(client, "op@example.com")
        r = client.get("/desk/privacy/export",
                       headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_export_has_required_keys(self, client):
        tenant = _make_tenant()
        _make_user(tenant)
        token = _login(client, "op@example.com")
        data = client.get("/desk/privacy/export",
                          headers={"Authorization": f"Bearer {token}"}).json()
        for key in ("tenant", "users", "messages", "drafts",
                    "audit_log", "usage_events", "approval_queue"):
            assert key in data, f"missing key: {key}"

    def test_export_includes_messages(self, client):
        tenant = _make_tenant()
        _make_user(tenant)
        _ingest(tenant.id, "exp-1", "Invoice request")
        token = _login(client, "op@example.com")
        data = client.get("/desk/privacy/export",
                          headers={"Authorization": f"Bearer {token}"}).json()
        assert len(data["messages"]) >= 1

    def test_export_is_tenant_scoped(self, client):
        t1 = _make_tenant("t1-exp")
        t2 = _make_tenant("t2-exp")
        _make_user(t1, "e1@example.com")
        _make_user(t2, "e2@example.com")
        _ingest(t2.id, "other-exp", "Tenant 2 message")
        token1 = _login(client, "e1@example.com")
        data = client.get("/desk/privacy/export",
                          headers={"Authorization": f"Bearer {token1}"}).json()
        assert len(data["messages"]) == 0

    def test_export_does_not_include_raw_json(self, client):
        tenant = _make_tenant()
        _make_user(tenant)
        _ingest(tenant.id, "raw-1", "Raw check")
        token = _login(client, "op@example.com")
        data = client.get("/desk/privacy/export",
                          headers={"Authorization": f"Bearer {token}"}).json()
        for msg in data["messages"]:
            assert "raw_json" not in msg, "raw_json must not appear in export by default"


# ---------------------------------------------------------------------------
# Erasure endpoint
# ---------------------------------------------------------------------------

class TestPrivacyErase:
    def test_erase_requires_auth(self, client):
        r = client.delete("/desk/privacy/data")
        assert r.status_code == 401

    def test_erase_returns_200(self, client):
        tenant = _make_tenant()
        _make_user(tenant)
        token = _login(client, "op@example.com")
        r = client.delete("/desk/privacy/data",
                          headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_erase_deletes_messages(self, client):
        tenant = _make_tenant()
        _make_user(tenant)
        _ingest(tenant.id, "del-1", "To be erased")
        token = _login(client, "op@example.com")
        result = client.delete("/desk/privacy/data",
                               headers={"Authorization": f"Bearer {token}"}).json()
        assert result["deleted_messages"] >= 1

    def test_erase_messages_gone_after(self, client):
        tenant = _make_tenant()
        _make_user(tenant)
        _ingest(tenant.id, "del-2", "Gone after erase")
        token = _login(client, "op@example.com")
        hdr = {"Authorization": f"Bearer {token}"}
        client.delete("/desk/privacy/data", headers=hdr)
        r = client.get("/desk/queue", headers=hdr)
        assert r.json()["count"] == 0

    def test_erase_skips_legal_hold(self, client):
        tenant = _make_tenant()
        _make_user(tenant)
        msg = _ingest(tenant.id, "hold-1", "Legal hold message")
        # Set legal_hold = True
        db = SessionLocal()
        try:
            row = db.query(Message).filter(Message.id == msg.id).first()
            row.legal_hold = True
            db.commit()
        finally:
            db.close()
        token = _login(client, "op@example.com")
        result = client.delete("/desk/privacy/data",
                               headers={"Authorization": f"Bearer {token}"}).json()
        assert result["legal_hold_skipped"] >= 1
        # Message still exists
        db = SessionLocal()
        try:
            count = db.query(Message).filter(
                Message.tenant_id == tenant.id
            ).count()
        finally:
            db.close()
        assert count >= 1

    def test_erase_is_tenant_scoped(self, client):
        """Erasing tenant 1 does not delete tenant 2's messages."""
        t1 = _make_tenant("t1-era")
        t2 = _make_tenant("t2-era")
        _make_user(t1, "era1@example.com")
        _make_user(t2, "era2@example.com")
        _ingest(t2.id, "t2-msg", "Tenant 2 message")
        token1 = _login(client, "era1@example.com")
        client.delete("/desk/privacy/data",
                      headers={"Authorization": f"Bearer {token1}"})
        db = SessionLocal()
        try:
            count = db.query(Message).filter(Message.tenant_id == t2.id).count()
        finally:
            db.close()
        assert count >= 1

    def test_erase_deletes_drafts(self, client):
        tenant = _make_tenant()
        _make_user(tenant)
        _make_draft(tenant.id)
        token = _login(client, "op@example.com")
        result = client.delete("/desk/privacy/data",
                               headers={"Authorization": f"Bearer {token}"}).json()
        assert result["deleted_drafts"] >= 1

    def test_erase_response_has_required_keys(self, client):
        tenant = _make_tenant()
        _make_user(tenant)
        token = _login(client, "op@example.com")
        result = client.delete("/desk/privacy/data",
                               headers={"Authorization": f"Bearer {token}"}).json()
        for key in ("tenant_id", "deleted_messages", "deleted_drafts",
                    "legal_hold_skipped"):
            assert key in result, f"missing key: {key}"


# ---------------------------------------------------------------------------
# apply_retention respects legal_hold
# ---------------------------------------------------------------------------

class TestRetentionLegalHold:
    def test_apply_retention_skips_legal_hold(self):
        """Messages with legal_hold=True are not deleted by apply_retention."""
        from datetime import timedelta
        tenant = _make_tenant()
        msg = _ingest(tenant.id, "ret-hold-1", "Old held message")
        # Set legal_hold=True and backdate ingest_time beyond retention window
        db = SessionLocal()
        try:
            row = db.query(Message).filter(Message.id == msg.id).first()
            row.legal_hold = True
            row.ingest_time = datetime.now(timezone.utc) - timedelta(days=400)
            db.commit()
        finally:
            db.close()
        db = SessionLocal()
        try:
            deleted = apply_retention(
                db, tenant.id, tenant_retention_days=180
            )
        finally:
            db.close()
        assert deleted == 0
        db = SessionLocal()
        try:
            count = db.query(Message).filter(Message.tenant_id == tenant.id).count()
        finally:
            db.close()
        assert count == 1

    def test_apply_retention_deletes_normal_old_messages(self):
        """Non-held messages older than the window are deleted."""
        from datetime import timedelta
        tenant = _make_tenant()
        msg = _ingest(tenant.id, "ret-old-1", "Old message")
        db = SessionLocal()
        try:
            row = db.query(Message).filter(Message.id == msg.id).first()
            row.ingest_time = datetime.now(timezone.utc) - timedelta(days=400)
            db.commit()
        finally:
            db.close()
        db = SessionLocal()
        try:
            deleted = apply_retention(db, tenant.id, tenant_retention_days=180)
        finally:
            db.close()
        assert deleted >= 1


# ---------------------------------------------------------------------------
# delete_tenant_data counts
# ---------------------------------------------------------------------------

class TestDeleteTenantData:
    def test_returns_expected_keys(self):
        tenant = _make_tenant()
        db = SessionLocal()
        try:
            result = delete_tenant_data(db, tenant.id)
        finally:
            db.close()
        for key in ("tenant_id", "deleted_messages", "deleted_drafts",
                    "deleted_audit_log", "deleted_usage_events",
                    "deleted_approval_queue", "legal_hold_skipped"):
            assert key in result, f"missing key: {key}"

    def test_counts_messages(self):
        tenant = _make_tenant()
        _ingest(tenant.id, "dtd-1", "Message 1")
        _ingest(tenant.id, "dtd-2", "Message 2")
        db = SessionLocal()
        try:
            result = delete_tenant_data(db, tenant.id)
        finally:
            db.close()
        assert result["deleted_messages"] == 2

    def test_counts_drafts(self):
        tenant = _make_tenant()
        _make_draft(tenant.id)
        _make_draft(tenant.id, "shadow")
        db = SessionLocal()
        try:
            result = delete_tenant_data(db, tenant.id)
        finally:
            db.close()
        assert result["deleted_drafts"] == 2

    def test_legal_hold_increments_skipped(self):
        tenant = _make_tenant()
        msg = _ingest(tenant.id, "dtd-hold", "Held message")
        db = SessionLocal()
        try:
            row = db.query(Message).filter(Message.id == msg.id).first()
            row.legal_hold = True
            db.commit()
        finally:
            db.close()
        db = SessionLocal()
        try:
            result = delete_tenant_data(db, tenant.id)
        finally:
            db.close()
        assert result["legal_hold_skipped"] == 1
        assert result["deleted_messages"] == 0


# ---------------------------------------------------------------------------
# _SecretFilter scrubs secrets from log records
# ---------------------------------------------------------------------------

class TestSecretFilter:
    def test_filter_scrubs_imap_password(self):
        """_SecretFilter must replace IMAP_PASSWORD value in log output."""
        from app.workers.intake_loop import _SecretFilter
        secret = "super-secret-pw-987"
        prev = os.environ.get("IMAP_PASSWORD")
        os.environ["IMAP_PASSWORD"] = secret
        try:
            filt = _SecretFilter()
            record = logging.LogRecord(
                name="test", level=logging.ERROR,
                pathname="", lineno=0,
                msg="login failed for %s",
                args=(secret,), exc_info=None,
            )
            filt.filter(record)
            # After filtering, the interpolated message should not contain the secret.
            assert secret not in record.getMessage()
        finally:
            if prev is None:
                os.environ.pop("IMAP_PASSWORD", None)
            else:
                os.environ["IMAP_PASSWORD"] = prev

    def test_filter_passes_non_secret_records(self):
        """Records not containing secret values pass through unchanged."""
        from app.workers.intake_loop import _SecretFilter
        filt = _SecretFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0,
            msg="tenant=%s processed=3",
            args=(1,), exc_info=None,
        )
        result = filt.filter(record)
        assert result is True


# ---------------------------------------------------------------------------
# Redaction before model call (regression guard)
# ---------------------------------------------------------------------------

class TestRedactionGuard:
    def test_email_redacted(self):
        text, count = redact("Contact us at user@example.com for details.")
        assert "[EMAIL]" in text
        assert "user@example.com" not in text
        assert count >= 1

    def test_iban_redacted(self):
        text, count = redact("Pay to DE89370400440532013000.")
        assert "[IBAN]" in text
        assert count >= 1

    def test_phone_redacted(self):
        text, count = redact("Call +49 89 12345678 now.")
        assert "[PHONE]" in text
        assert count >= 1

    def test_redact_message_covers_subject_and_body(self):
        rs, rb, total = redact_message(
            "Invoice from user@example.com",
            "Call +49 30 12345678 for support.",
        )
        assert "user@example.com" not in rs
        assert "[PHONE]" in rb
        assert total >= 2

    def test_clean_text_has_zero_count(self):
        _, count = redact("No personal data here.")
        assert count == 0
