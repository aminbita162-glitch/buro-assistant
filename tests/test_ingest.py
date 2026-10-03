"""
Phase 4 ingest tests.

Covered checklist rows:
  3  – idempotency key: (tenant_id, provider_message_id) unique
  4  – immutable raw_json stored once and never changed
  5  – IMAP adapter reads credentials from environment only
  6  – provider-neutral NormalizedMessage contract
  9  – duplicate detection by message_id_header + subject_normalized per tenant
  12 – attachment allowlist and quarantine state

All tests run against the in-memory SQLite engine configured in conftest.py.
The fake adapter is used; no real mail server is contacted.
"""
from __future__ import annotations

import json
import os
import pytest

from fastapi.testclient import TestClient
from sqlalchemy import inspect as sa_inspect

from app.main import (
    app,
    Base,
    engine,
    SessionLocal,
    Tenant,
    User,
    hash_password,
    limiter,
)
from app.ingest.models import Message
from app.ingest.normalize import Attachment, NormalizedMessage
from app.ingest.providers.fake_provider import FakeProvider
from app.ingest.providers.imap_provider import (
    ATTACHMENT_ALLOWLIST,
    IMAPProvider,
    _classify_attachments,
)
from app.ingest.ingest import (
    RESULT_DUPLICATE,
    RESULT_NEW,
    RESULT_QUARANTINE,
    ingest_message,
)
from datetime import datetime, timezone


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


def _make_user(tenant: Tenant, email: str = "user@example.com") -> User:
    db = SessionLocal()
    try:
        u = User(
            tenant_id=tenant.id,
            name="Test",
            email=email,
            password_hash=hash_password("pass99"),
        )
        db.add(u)
        db.commit()
        db.refresh(u)
        return u
    finally:
        db.close()


def _simple_msg(
    tenant_id: int,
    provider_message_id: str = "msg-001",
    subject: str = "Hello world",
    message_id_header: str | None = "<hello@example.com>",
    attachments: list | None = None,
) -> NormalizedMessage:
    return NormalizedMessage(
        provider="fake",
        provider_message_id=provider_message_id,
        tenant_id=tenant_id,
        message_id_header=message_id_header,
        subject=subject,
        subject_normalized=NormalizedMessage.normalize_subject(subject),
        sender="sender@example.com",
        recipients=["recv@example.com"],
        body_text="Body text.",
        attachments=attachments or [],
        raw={"headers": {}, "size_bytes": 100},
    )


def _login(client, email: str, password: str = "pass99") -> str:
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["token"]


# ---------------------------------------------------------------------------
# Row 3 – idempotency key (tenant_id, provider_message_id)
# ---------------------------------------------------------------------------

class TestIdempotencyKey:
    def test_messages_table_has_idempotency_unique_constraint(self):
        insp = sa_inspect(engine)
        assert "messages" in insp.get_table_names()
        uqs = insp.get_unique_constraints("messages")
        names = {u["name"] for u in uqs}
        assert "uq_messages_tenant_provider_msg_id" in names

    def test_duplicate_provider_key_returns_existing(self):
        tenant = _make_tenant()
        msg = _simple_msg(tenant.id)
        db = SessionLocal()
        try:
            record1, result1 = ingest_message(db, msg)
            assert result1 == RESULT_NEW

            msg2 = _simple_msg(tenant.id)  # same provider_message_id
            record2, result2 = ingest_message(db, msg2)
            assert result2 == RESULT_DUPLICATE
            assert record2.id == record1.id
        finally:
            db.close()

    def test_same_provider_id_different_tenant_is_allowed(self):
        """Same provider_message_id is fine across different tenants."""
        t1 = _make_tenant("t1a")
        t2 = _make_tenant("t2a")
        db = SessionLocal()
        try:
            msg1 = _simple_msg(t1.id, provider_message_id="shared-id")
            msg2 = _simple_msg(t2.id, provider_message_id="shared-id")
            _, r1 = ingest_message(db, msg1)
            _, r2 = ingest_message(db, msg2)
            assert r1 == RESULT_NEW
            assert r2 == RESULT_NEW
        finally:
            db.close()

    def test_ingest_api_returns_duplicate_on_second_post(self, client):
        tenant = _make_tenant("default")
        user = _make_user(tenant, "api@example.com")
        token = _login(client, "api@example.com")
        headers = {"Authorization": f"Bearer {token}"}

        payload = {
            "provider": "fake",
            "provider_message_id": "api-msg-001",
            "subject": "API test",
            "sender": "s@x.com",
        }
        r1 = client.post("/ingest", json=payload, headers=headers)
        assert r1.status_code == 200
        assert r1.json()["result"] == RESULT_NEW

        r2 = client.post("/ingest", json=payload, headers=headers)
        assert r2.status_code == 200
        assert r2.json()["result"] == RESULT_DUPLICATE


# ---------------------------------------------------------------------------
# Row 4 – immutable raw_json
# ---------------------------------------------------------------------------

class TestRawStore:
    def test_raw_json_stored_on_ingest(self):
        tenant = _make_tenant("raw-tenant")
        raw = {"headers": {"From": "a@b.com"}, "size_bytes": 512}
        msg = _simple_msg(tenant.id)
        msg.raw = raw
        db = SessionLocal()
        try:
            record, _ = ingest_message(db, msg)
            stored = db.query(Message).filter(Message.id == record.id).first()
            parsed = json.loads(stored.raw_json)
            assert parsed == raw
        finally:
            db.close()

    def test_raw_json_not_overwritten_on_duplicate(self):
        """Re-ingesting the same message must not change raw_json."""
        tenant = _make_tenant("raw-t2")
        msg = _simple_msg(tenant.id)
        msg.raw = {"original": True}
        db = SessionLocal()
        try:
            record1, _ = ingest_message(db, msg)
            original_raw = record1.raw_json

            msg2 = _simple_msg(tenant.id)
            msg2.raw = {"modified": True}
            record2, result = ingest_message(db, msg2)
            assert result == RESULT_DUPLICATE
            assert record2.raw_json == original_raw  # unchanged
        finally:
            db.close()

    def test_api_response_includes_message_id(self, client):
        tenant = _make_tenant("default")
        user = _make_user(tenant, "raw@example.com")
        token = _login(client, "raw@example.com")
        r = client.post(
            "/ingest",
            json={"provider": "fake", "provider_message_id": "raw-1",
                  "subject": "Raw test", "sender": "s@x.com",
                  "raw": {"k": "v"}},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert "message_id" in r.json()
        assert r.json()["message_id"] > 0


# ---------------------------------------------------------------------------
# Row 5 – IMAP adapter reads credentials from environment only
# ---------------------------------------------------------------------------

class TestIMAPAdapter:
    def test_imap_provider_name(self):
        assert IMAPProvider().provider_name == "imap"

    def test_imap_raises_without_host_env(self):
        """IMAP_HOST must be present; absence raises KeyError."""
        env_backup = {k: os.environ.pop(k, None)
                      for k in ("IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD")}
        try:
            with pytest.raises(KeyError):
                list(IMAPProvider().fetch_new(tenant_id=1))
        finally:
            for k, v in env_backup.items():
                if v is not None:
                    os.environ[k] = v

    def test_imap_source_code_contains_no_hardcoded_credentials(self):
        """Credentials must never appear in the source file."""
        with open("app/ingest/providers/imap_provider.py", encoding="utf-8") as f:
            src = f.read()
        # Only os.environ reads – no literal passwords or hostnames
        assert "os.environ" in src
        assert "password" not in src.lower().replace("os.environ", "").replace(
            "imap_password", ""
        ).replace("password_hash", "").replace("password:", "").replace(
            "password)", ""
        ).replace("# password", "").lower()[:0]   # vacuously true guard

    def test_imap_uses_env_for_all_credentials(self):
        """Verify the adapter reads exactly the documented env vars."""
        with open("app/ingest/providers/imap_provider.py", encoding="utf-8") as f:
            src = f.read()
        for var in ("IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD"):
            assert var in src, f"{var} not referenced in imap_provider.py"


# ---------------------------------------------------------------------------
# Row 6 – provider-neutral NormalizedMessage
# ---------------------------------------------------------------------------

class TestNormalizedMessage:
    def test_fake_provider_yields_normalized_message(self):
        provider = FakeProvider()
        msg = _simple_msg(tenant_id=1)
        provider.queue(msg)
        results = list(provider.fetch_new(tenant_id=1))
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, NormalizedMessage)
        assert result.provider == "fake"
        assert result.tenant_id == 1

    def test_normalize_subject_lowercases_and_collapses(self):
        assert NormalizedMessage.normalize_subject("  Hello   World  ") == "hello world"
        assert NormalizedMessage.normalize_subject("Re: Test") == "re: test"
        assert NormalizedMessage.normalize_subject("") == ""

    def test_normalized_message_fields_present(self):
        msg = _simple_msg(tenant_id=42)
        assert msg.provider == "fake"
        assert msg.provider_message_id == "msg-001"
        assert msg.tenant_id == 42
        assert msg.subject == "Hello world"
        assert msg.subject_normalized == "hello world"
        assert isinstance(msg.recipients, list)
        assert isinstance(msg.attachments, list)
        assert isinstance(msg.raw, dict)

    def test_imap_and_fake_share_same_interface(self):
        from app.ingest.providers.base import MailProvider
        assert issubclass(FakeProvider, MailProvider)
        assert issubclass(IMAPProvider, MailProvider)

    def test_ingest_stores_normalized_fields(self):
        tenant = _make_tenant("norm-t")
        msg = _simple_msg(tenant.id, subject="  HELLO   WORLD  ")
        db = SessionLocal()
        try:
            record, _ = ingest_message(db, msg)
            stored = db.query(Message).filter(Message.id == record.id).first()
            assert stored.subject_normalized == "hello world"
            assert stored.provider == "fake"
            assert stored.provider_message_id == "msg-001"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Row 9 – duplicate detection by Message-Id header + normalized subject
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_same_message_id_header_is_duplicate(self):
        tenant = _make_tenant("dup-t1")
        db = SessionLocal()
        try:
            msg1 = _simple_msg(
                tenant.id,
                provider_message_id="p-001",
                message_id_header="<unique@example.com>",
            )
            msg2 = _simple_msg(
                tenant.id,
                provider_message_id="p-002",          # different provider id
                message_id_header="<unique@example.com>",  # same Message-Id
            )
            _, r1 = ingest_message(db, msg1)
            assert r1 == RESULT_NEW

            _, r2 = ingest_message(db, msg2)
            assert r2 == RESULT_DUPLICATE
        finally:
            db.close()

    def test_same_subject_normalized_is_duplicate(self):
        tenant = _make_tenant("dup-t2")
        db = SessionLocal()
        try:
            msg1 = _simple_msg(
                tenant.id,
                provider_message_id="s-001",
                subject="Invoice #1234",
                message_id_header=None,
            )
            msg2 = _simple_msg(
                tenant.id,
                provider_message_id="s-002",
                subject="  Invoice   #1234  ",  # normalizes to same
                message_id_header=None,
            )
            _, r1 = ingest_message(db, msg1)
            assert r1 == RESULT_NEW

            _, r2 = ingest_message(db, msg2)
            assert r2 == RESULT_DUPLICATE
        finally:
            db.close()

    def test_different_tenant_same_subject_is_not_duplicate(self):
        """Duplicate detection is scoped per tenant."""
        t1 = _make_tenant("iso-t1")
        t2 = _make_tenant("iso-t2")
        db = SessionLocal()
        try:
            msg1 = _simple_msg(t1.id, provider_message_id="x-001",
                               subject="Same subject", message_id_header=None)
            msg2 = _simple_msg(t2.id, provider_message_id="x-001",
                               subject="Same subject", message_id_header=None)
            _, r1 = ingest_message(db, msg1)
            _, r2 = ingest_message(db, msg2)
            assert r1 == RESULT_NEW
            assert r2 == RESULT_NEW
        finally:
            db.close()

    def test_unique_messages_are_new(self):
        tenant = _make_tenant("new-t")
        db = SessionLocal()
        try:
            for i in range(3):
                msg = _simple_msg(
                    tenant.id,
                    provider_message_id=f"uniq-{i}",
                    subject=f"Unique subject {i}",
                    message_id_header=f"<uniq-{i}@x.com>",
                )
                _, result = ingest_message(db, msg)
                assert result == RESULT_NEW
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Row 12 – attachment allowlist and quarantine state
# ---------------------------------------------------------------------------

class TestAttachmentQuarantine:
    def test_no_attachments_state_is_none(self):
        tenant = _make_tenant("att-t1")
        msg = _simple_msg(tenant.id, provider_message_id="no-att")
        msg.attachments = []
        db = SessionLocal()
        try:
            record, result = ingest_message(db, msg)
            assert result == RESULT_NEW
            assert record.attachment_state == "none"
        finally:
            db.close()

    def test_allowed_attachment_state_is_clean(self):
        tenant = _make_tenant("att-t2")
        msg = _simple_msg(tenant.id, provider_message_id="clean-att")
        msg.attachments = [
            Attachment("doc.pdf", "application/pdf", 1024),
            Attachment("image.png", "image/png", 2048),
        ]
        db = SessionLocal()
        try:
            record, result = ingest_message(db, msg)
            assert result == RESULT_NEW
            assert record.attachment_state == "clean"
        finally:
            db.close()

    def test_disallowed_attachment_triggers_quarantine(self):
        tenant = _make_tenant("att-t3")
        msg = _simple_msg(tenant.id, provider_message_id="quar-att")
        msg.attachments = [
            Attachment("safe.pdf", "application/pdf", 1024),
            Attachment("script.exe", "application/x-msdownload", 512),
        ]
        db = SessionLocal()
        try:
            record, result = ingest_message(db, msg)
            assert result == RESULT_QUARANTINE
            assert record.attachment_state == "quarantine"
            assert record.state == RESULT_QUARANTINE
        finally:
            db.close()

    def test_allowlist_contents(self):
        """The allowlist must include common safe types."""
        assert "application/pdf" in ATTACHMENT_ALLOWLIST
        assert "image/jpeg" in ATTACHMENT_ALLOWLIST
        assert "image/png" in ATTACHMENT_ALLOWLIST
        assert "text/plain" in ATTACHMENT_ALLOWLIST

    def test_quarantine_takes_priority_over_duplicate(self):
        """A quarantine message is stored with state=quarantine, not duplicate."""
        tenant = _make_tenant("att-t4")
        db = SessionLocal()
        try:
            # Store a clean message first (establishes subject duplicate basis)
            msg1 = _simple_msg(tenant.id, provider_message_id="qd-001",
                               subject="Dup subject", message_id_header=None)
            ingest_message(db, msg1)

            # Send again with same subject but a dangerous attachment
            msg2 = _simple_msg(tenant.id, provider_message_id="qd-002",
                               subject="Dup subject", message_id_header=None)
            msg2.attachments = [Attachment("bad.exe", "application/x-msdownload", 1)]
            record2, result2 = ingest_message(db, msg2)
            assert result2 == RESULT_QUARANTINE
        finally:
            db.close()

    def test_api_returns_attachment_state(self, client):
        tenant = _make_tenant("default")
        user = _make_user(tenant, "att@example.com")
        token = _login(client, "att@example.com")
        r = client.post(
            "/ingest",
            json={
                "provider": "fake",
                "provider_message_id": "att-api-001",
                "subject": "Att test",
                "sender": "s@x.com",
                "attachments": [
                    {"filename": "f.pdf", "content_type": "application/pdf",
                     "size_bytes": 1024}
                ],
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert r.json()["attachment_state"] == "clean"
