"""
tests/test_sender_auth.py – Phase 1: sender authentication tests.

Covers:
  - SenderAuthResult: pass, fail, not_run values and any_fail property
  - check_sender_auth returns not_run when SENDER_AUTH_ENABLED is unset
  - check_sender_auth returns not_run for bad sender inputs
  - Ingest stores auth_spf / auth_dkim / auth_dmarc on the Message row
  - Pre-populated auth results on NormalizedMessage are stored as-is
  - A fail result does not delete the message (message still exists)
  - A fail result blocks auto-send (pipeline outcome is "draft", not "send")
  - A pass result does not block auto-send
  - A not_run result does not block auto-send
  - Triage decision includes auth fields readable by Amin
  - DNS check path: mock resolver returns pass / fail / not_run per case

All tests use in-memory SQLite and the fake provider.  No real DNS is called.
"""
from __future__ import annotations

import os
import sys
import types
import pytest
from unittest.mock import patch, MagicMock

from app.main import Base, engine, SessionLocal, Tenant, limiter
from app.ingest.normalize import NormalizedMessage, Attachment
from app.ingest.models import Message
from app.ingest.ingest import ingest_message, RESULT_NEW
from app.ingest.sender_auth import (
    SenderAuthResult,
    check_sender_auth,
    AUTH_PASS,
    AUTH_FAIL,
    AUTH_NOT_RUN,
    NOT_RUN_RESULT,
)
from app.pipeline import run_pipeline
from app.agents.fake_model import FakeModel


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


@pytest.fixture(autouse=True)
def clear_auth_env():
    """Ensure SENDER_AUTH_ENABLED is unset unless a test sets it explicitly."""
    os.environ.pop("SENDER_AUTH_ENABLED", None)
    yield
    os.environ.pop("SENDER_AUTH_ENABLED", None)


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


def _simple_msg(
    tenant_id: int,
    provider_message_id: str = "msg-auth-001",
    sender: str = "user@example.com",
    auth_spf: str = "not_run",
    auth_dkim: str = "not_run",
    auth_dmarc: str = "not_run",
) -> NormalizedMessage:
    msg = NormalizedMessage(
        provider="fake",
        provider_message_id=provider_message_id,
        tenant_id=tenant_id,
        message_id_header=None,
        subject="Auth test",
        subject_normalized="auth test",
        sender=sender,
        recipients=["desk@co.com"],
        body_text="Auth body.",
        attachments=[],
        raw={"headers": {}},
    )
    msg.auth_spf = auth_spf
    msg.auth_dkim = auth_dkim
    msg.auth_dmarc = auth_dmarc
    return msg


_RULE_PACK = {
    "subject_rules": [
        {"match": "invoice", "department": "finance", "action": "draft_reply"},
    ],
}

_SEND_POLICY = {"auto_reply_enabled": True}


# ---------------------------------------------------------------------------
# SenderAuthResult unit tests
# ---------------------------------------------------------------------------

class TestSenderAuthResult:
    def test_default_is_not_run(self):
        r = SenderAuthResult()
        assert r.spf == AUTH_NOT_RUN
        assert r.dkim == AUTH_NOT_RUN
        assert r.dmarc == AUTH_NOT_RUN

    def test_pass_values(self):
        r = SenderAuthResult(spf=AUTH_PASS, dkim=AUTH_PASS, dmarc=AUTH_PASS)
        assert r.spf == AUTH_PASS
        assert r.dkim == AUTH_PASS
        assert r.dmarc == AUTH_PASS

    def test_fail_values(self):
        r = SenderAuthResult(spf=AUTH_FAIL, dkim=AUTH_FAIL, dmarc=AUTH_FAIL)
        assert r.spf == AUTH_FAIL
        assert r.dkim == AUTH_FAIL
        assert r.dmarc == AUTH_FAIL

    def test_any_fail_true_on_single_fail(self):
        assert SenderAuthResult(spf=AUTH_FAIL).any_fail is True
        assert SenderAuthResult(dkim=AUTH_FAIL).any_fail is True
        assert SenderAuthResult(dmarc=AUTH_FAIL).any_fail is True

    def test_any_fail_false_when_no_fail(self):
        assert SenderAuthResult(spf=AUTH_PASS, dkim=AUTH_PASS, dmarc=AUTH_PASS).any_fail is False
        assert SenderAuthResult().any_fail is False

    def test_any_fail_mixed_pass_and_not_run(self):
        r = SenderAuthResult(spf=AUTH_PASS, dkim=AUTH_NOT_RUN, dmarc=AUTH_PASS)
        assert r.any_fail is False

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            SenderAuthResult(spf="unknown")

    def test_not_run_result_sentinel(self):
        assert NOT_RUN_RESULT.spf == AUTH_NOT_RUN
        assert NOT_RUN_RESULT.dkim == AUTH_NOT_RUN
        assert NOT_RUN_RESULT.dmarc == AUTH_NOT_RUN
        assert NOT_RUN_RESULT.any_fail is False


# ---------------------------------------------------------------------------
# check_sender_auth: not_run when SENDER_AUTH_ENABLED is off
# ---------------------------------------------------------------------------

class TestCheckSenderAuthDisabled:
    def test_returns_not_run_when_disabled(self):
        """Default: SENDER_AUTH_ENABLED is unset → all not_run."""
        result = check_sender_auth("user@example.com")
        assert result.spf == AUTH_NOT_RUN
        assert result.dkim == AUTH_NOT_RUN
        assert result.dmarc == AUTH_NOT_RUN

    def test_returns_not_run_for_empty_sender(self):
        result = check_sender_auth("")
        assert result == NOT_RUN_RESULT

    def test_returns_not_run_for_sender_without_at(self):
        result = check_sender_auth("not-an-email")
        assert result == NOT_RUN_RESULT

    def test_returns_not_run_when_enabled_but_dnspython_absent(self):
        """When dnspython is not installed, every check falls back to not_run."""
        os.environ["SENDER_AUTH_ENABLED"] = "1"
        # Patch builtins.__import__ to simulate ImportError for dns
        import builtins
        original_import = builtins.__import__

        def _no_dns(name, *args, **kwargs):
            if name == "dns.resolver" or name == "dns":
                raise ImportError(f"No module named {name!r}")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=_no_dns):
            result = check_sender_auth("user@example.com")

        # All three must be not_run (ImportError path)
        assert result.spf == AUTH_NOT_RUN
        assert result.dkim == AUTH_NOT_RUN
        assert result.dmarc == AUTH_NOT_RUN


# ---------------------------------------------------------------------------
# check_sender_auth: DNS mock — pass path
# ---------------------------------------------------------------------------

class TestCheckSenderAuthPass:
    def _make_dns_pass(self):
        """Build a mock dns.resolver module that returns passing records."""
        mock_dns = types.ModuleType("dns")
        mock_resolver = types.ModuleType("dns.resolver")

        def resolve(name, rtype, lifetime=5):
            name = str(name)
            if rtype == "TXT":
                rdata = MagicMock()
                if name.startswith("_dmarc."):
                    rdata.strings = [b"v=DMARC1; p=none"]
                elif name.endswith("._domainkey.example.com"):
                    rdata.strings = [b"v=DKIM1; k=rsa; p=AAABBB"]
                else:
                    # SPF
                    rdata.strings = [b"v=spf1 include:example.com ~all"]
                return [rdata]
            raise Exception("unknown rtype")

        mock_resolver.resolve = resolve
        mock_dns.resolver = mock_resolver
        return mock_dns, mock_resolver

    def test_spf_pass(self):
        os.environ["SENDER_AUTH_ENABLED"] = "1"
        mock_dns, mock_resolver = self._make_dns_pass()
        with patch.dict(sys.modules, {"dns": mock_dns, "dns.resolver": mock_resolver}):
            result = check_sender_auth("user@example.com")
        assert result.spf == AUTH_PASS

    def test_dmarc_pass(self):
        os.environ["SENDER_AUTH_ENABLED"] = "1"
        mock_dns, mock_resolver = self._make_dns_pass()
        with patch.dict(sys.modules, {"dns": mock_dns, "dns.resolver": mock_resolver}):
            result = check_sender_auth("user@example.com")
        assert result.dmarc == AUTH_PASS

    def test_dkim_pass_when_signature_header_present(self):
        os.environ["SENDER_AUTH_ENABLED"] = "1"
        mock_dns, mock_resolver = self._make_dns_pass()
        headers = {
            "DKIM-Signature": "v=1; a=rsa-sha256; d=example.com; s=selector1; bh=abc; b=xyz"
        }
        with patch.dict(sys.modules, {"dns": mock_dns, "dns.resolver": mock_resolver}):
            result = check_sender_auth("user@example.com", raw_headers=headers)
        assert result.dkim == AUTH_PASS

    def test_all_pass(self):
        os.environ["SENDER_AUTH_ENABLED"] = "1"
        mock_dns, mock_resolver = self._make_dns_pass()
        headers = {
            "DKIM-Signature": "v=1; a=rsa-sha256; d=example.com; s=selector1; bh=abc; b=xyz"
        }
        with patch.dict(sys.modules, {"dns": mock_dns, "dns.resolver": mock_resolver}):
            result = check_sender_auth("user@example.com", raw_headers=headers)
        assert result.spf == AUTH_PASS
        assert result.dkim == AUTH_PASS
        assert result.dmarc == AUTH_PASS
        assert result.any_fail is False


# ---------------------------------------------------------------------------
# check_sender_auth: DNS mock — fail path
# ---------------------------------------------------------------------------

class TestCheckSenderAuthFail:
    def _make_dns_fail(self):
        """Build a mock dns.resolver that returns failing SPF and no DMARC."""
        mock_dns = types.ModuleType("dns")
        mock_resolver = types.ModuleType("dns.resolver")

        def resolve(name, rtype, lifetime=5):
            name = str(name)
            if rtype == "TXT":
                rdata = MagicMock()
                if name.startswith("_dmarc."):
                    raise Exception("NXDOMAIN")
                elif name.endswith("._domainkey.example.com"):
                    raise Exception("NXDOMAIN")
                else:
                    # SPF hard-fail
                    rdata.strings = [b"v=spf1 -all"]
                return [rdata]
            raise Exception("unknown rtype")

        mock_resolver.resolve = resolve
        mock_dns.resolver = mock_resolver
        return mock_dns, mock_resolver

    def test_spf_fail_on_hard_fail_record(self):
        os.environ["SENDER_AUTH_ENABLED"] = "1"
        mock_dns, mock_resolver = self._make_dns_fail()
        with patch.dict(sys.modules, {"dns": mock_dns, "dns.resolver": mock_resolver}):
            result = check_sender_auth("user@example.com")
        assert result.spf == AUTH_FAIL

    def test_dkim_fail_when_no_signature_header(self):
        os.environ["SENDER_AUTH_ENABLED"] = "1"
        mock_dns, mock_resolver = self._make_dns_fail()
        with patch.dict(sys.modules, {"dns": mock_dns, "dns.resolver": mock_resolver}):
            result = check_sender_auth("user@example.com", raw_headers={})
        assert result.dkim == AUTH_FAIL

    def test_dmarc_not_run_when_lookup_raises(self):
        """DMARC raises an unexpected exception → not_run (safe fallback)."""
        os.environ["SENDER_AUTH_ENABLED"] = "1"
        mock_dns, mock_resolver = self._make_dns_fail()
        with patch.dict(sys.modules, {"dns": mock_dns, "dns.resolver": mock_resolver}):
            result = check_sender_auth("user@example.com")
        # DMARC lookup raised → not_run (not fail)
        assert result.dmarc == AUTH_NOT_RUN

    def test_any_fail_true_when_spf_fails(self):
        os.environ["SENDER_AUTH_ENABLED"] = "1"
        mock_dns, mock_resolver = self._make_dns_fail()
        with patch.dict(sys.modules, {"dns": mock_dns, "dns.resolver": mock_resolver}):
            result = check_sender_auth("user@example.com")
        assert result.any_fail is True


# ---------------------------------------------------------------------------
# Ingest stores auth columns on Message row
# ---------------------------------------------------------------------------

class TestIngestStoresAuthColumns:
    def test_pre_populated_pass_stored(self):
        tenant = _make_tenant("auth-pass-t")
        msg = _simple_msg(tenant.id, auth_spf="pass", auth_dkim="pass", auth_dmarc="pass")
        db = SessionLocal()
        try:
            record, result = ingest_message(db, msg)
            stored = db.query(Message).filter(Message.id == record.id).first()
            assert stored.auth_spf == "pass"
            assert stored.auth_dkim == "pass"
            assert stored.auth_dmarc == "pass"
        finally:
            db.close()

    def test_pre_populated_fail_stored(self):
        tenant = _make_tenant("auth-fail-t")
        msg = _simple_msg(tenant.id, auth_spf="fail", auth_dkim="fail", auth_dmarc="fail")
        db = SessionLocal()
        try:
            record, result = ingest_message(db, msg)
            stored = db.query(Message).filter(Message.id == record.id).first()
            assert stored.auth_spf == "fail"
            assert stored.auth_dkim == "fail"
            assert stored.auth_dmarc == "fail"
        finally:
            db.close()

    def test_default_not_run_stored(self):
        """When SENDER_AUTH_ENABLED is off and no pre-populated values, all not_run."""
        tenant = _make_tenant("auth-norun-t")
        msg = _simple_msg(tenant.id)
        db = SessionLocal()
        try:
            record, result = ingest_message(db, msg)
            stored = db.query(Message).filter(Message.id == record.id).first()
            assert stored.auth_spf == "not_run"
            assert stored.auth_dkim == "not_run"
            assert stored.auth_dmarc == "not_run"
        finally:
            db.close()

    def test_message_has_auth_spf_column(self):
        """The messages table must have the three auth columns after schema creation."""
        from sqlalchemy import inspect as sa_inspect
        insp = sa_inspect(engine)
        cols = {c["name"] for c in insp.get_columns("messages")}
        assert "auth_spf" in cols
        assert "auth_dkim" in cols
        assert "auth_dmarc" in cols

    def test_fail_does_not_delete_message(self):
        """A fail result must keep the message in the DB (not delete it)."""
        tenant = _make_tenant("auth-nodelete-t")
        msg = _simple_msg(
            tenant.id,
            provider_message_id="fail-nodelete",
            auth_spf="fail",
            auth_dkim="fail",
            auth_dmarc="fail",
        )
        db = SessionLocal()
        try:
            record, result = ingest_message(db, msg)
            # Message must still be in the DB.
            stored = db.query(Message).filter(Message.id == record.id).first()
            assert stored is not None
            assert stored.auth_spf == "fail"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Pipeline: auth-fail blocks auto-send; message is preserved
# ---------------------------------------------------------------------------

class TestPipelineAuthSendBlock:
    """
    The pipeline must not produce outcome="send" when any auth result is "fail",
    even when auto_reply_enabled=True is set.
    """

    def _make_tenant_db(self, slug: str):
        tenant = _make_tenant(slug)
        db = SessionLocal()
        return tenant, db

    def test_auth_fail_blocks_send(self):
        """SPF fail → pipeline routes to draft, not send."""
        tenant, db = self._make_tenant_db("pipe-fail-t")
        msg = _simple_msg(
            tenant.id,
            provider_message_id="pipe-fail-001",
            sender="user@example.com",
            auth_spf="fail",
            auth_dkim="not_run",
            auth_dmarc="not_run",
        )
        msg.subject = "Invoice please"
        msg.subject_normalized = "invoice please"
        try:
            result = run_pipeline(
                db, msg,
                rule_pack=_RULE_PACK,
                policy_config=_SEND_POLICY,
                triage_model=None,
            )
            # Must NOT be "send" — auth fail blocks it.
            assert result.outcome != "send", (
                f"Expected outcome != 'send', got {result.outcome!r}"
            )
            # Message must be preserved (not deleted).
            assert result.message is not None
            stored = db.query(Message).filter(Message.id == result.message.id).first()
            assert stored is not None
        finally:
            db.close()

    def test_dkim_fail_blocks_send(self):
        """DKIM fail → pipeline does not send."""
        tenant, db = self._make_tenant_db("pipe-dkim-t")
        msg = _simple_msg(
            tenant.id,
            provider_message_id="pipe-dkim-001",
            sender="user@example.com",
            auth_spf="not_run",
            auth_dkim="fail",
            auth_dmarc="not_run",
        )
        msg.subject = "Invoice please"
        msg.subject_normalized = "invoice please"
        try:
            result = run_pipeline(
                db, msg,
                rule_pack=_RULE_PACK,
                policy_config=_SEND_POLICY,
                triage_model=None,
            )
            assert result.outcome != "send"
        finally:
            db.close()

    def test_dmarc_fail_blocks_send(self):
        """DMARC fail → pipeline does not send."""
        tenant, db = self._make_tenant_db("pipe-dmarc-t")
        msg = _simple_msg(
            tenant.id,
            provider_message_id="pipe-dmarc-001",
            sender="user@example.com",
            auth_spf="not_run",
            auth_dkim="not_run",
            auth_dmarc="fail",
        )
        msg.subject = "Invoice please"
        msg.subject_normalized = "invoice please"
        try:
            result = run_pipeline(
                db, msg,
                rule_pack=_RULE_PACK,
                policy_config=_SEND_POLICY,
                triage_model=None,
            )
            assert result.outcome != "send"
        finally:
            db.close()

    def test_auth_pass_does_not_block_send(self):
        """All auth pass → pipeline may send when policy allows."""
        tenant, db = self._make_tenant_db("pipe-pass-t")
        msg = _simple_msg(
            tenant.id,
            provider_message_id="pipe-pass-001",
            sender="user@example.com",
            auth_spf="pass",
            auth_dkim="pass",
            auth_dmarc="pass",
        )
        msg.subject = "Invoice please"
        msg.subject_normalized = "invoice please"
        try:
            result = run_pipeline(
                db, msg,
                rule_pack=_RULE_PACK,
                policy_config=_SEND_POLICY,
                triage_model=None,
            )
            assert result.outcome == "send"
        finally:
            db.close()

    def test_auth_not_run_does_not_block_send(self):
        """All not_run → pipeline may send when policy allows."""
        tenant, db = self._make_tenant_db("pipe-notrun-t")
        msg = _simple_msg(
            tenant.id,
            provider_message_id="pipe-notrun-001",
            sender="user@example.com",
            auth_spf="not_run",
            auth_dkim="not_run",
            auth_dmarc="not_run",
        )
        msg.subject = "Invoice please"
        msg.subject_normalized = "invoice please"
        try:
            result = run_pipeline(
                db, msg,
                rule_pack=_RULE_PACK,
                policy_config=_SEND_POLICY,
                triage_model=None,
            )
            assert result.outcome == "send"
        finally:
            db.close()

    def test_auth_fail_message_preserved_not_deleted(self):
        """A fail result must not delete the message — it must survive in the DB."""
        tenant, db = self._make_tenant_db("pipe-preserve-t")
        msg = _simple_msg(
            tenant.id,
            provider_message_id="pipe-preserve-001",
            sender="user@example.com",
            auth_spf="fail",
            auth_dkim="fail",
            auth_dmarc="fail",
        )
        msg.subject = "Invoice please"
        msg.subject_normalized = "invoice please"
        try:
            result = run_pipeline(
                db, msg,
                rule_pack=_RULE_PACK,
                policy_config=_SEND_POLICY,
                triage_model=None,
            )
            # Message must still exist.
            stored = db.query(Message).filter(Message.id == result.message.id).first()
            assert stored is not None
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Triage decision includes auth fields
# ---------------------------------------------------------------------------

class TestTriageDecisionIncludesAuth:
    def test_triage_decision_has_auth_fields(self):
        """The triage decision dict returned by Amin must include auth_spf/dkim/dmarc."""
        from app.agents.amin import triage
        msg = NormalizedMessage(
            provider="fake",
            provider_message_id="triage-auth-001",
            tenant_id=1,
            message_id_header=None,
            subject="Invoice please",
            subject_normalized="invoice please",
            sender="user@example.com",
            recipients=["desk@co.com"],
            body_text="Body.",
            attachments=[],
            raw={},
        )
        msg.auth_spf = "pass"
        msg.auth_dkim = "not_run"
        msg.auth_dmarc = "fail"
        decision = triage(msg, rule_pack=_RULE_PACK)
        assert "auth_spf" in decision
        assert "auth_dkim" in decision
        assert "auth_dmarc" in decision
        assert decision["auth_spf"] == "pass"
        assert decision["auth_dkim"] == "not_run"
        assert decision["auth_dmarc"] == "fail"

    def test_triage_decision_auth_pass(self):
        from app.agents.amin import triage
        msg = NormalizedMessage(
            provider="fake",
            provider_message_id="triage-auth-pass",
            tenant_id=1,
            message_id_header=None,
            subject="Invoice please",
            subject_normalized="invoice please",
            sender="user@example.com",
            recipients=["desk@co.com"],
            body_text="Body.",
            attachments=[],
            raw={},
        )
        msg.auth_spf = "pass"
        msg.auth_dkim = "pass"
        msg.auth_dmarc = "pass"
        decision = triage(msg, rule_pack=_RULE_PACK)
        assert decision["auth_spf"] == "pass"
        assert decision["auth_dkim"] == "pass"
        assert decision["auth_dmarc"] == "pass"

    def test_triage_decision_auth_not_run(self):
        from app.agents.amin import triage
        msg = NormalizedMessage(
            provider="fake",
            provider_message_id="triage-auth-notrun",
            tenant_id=1,
            message_id_header=None,
            subject="Invoice please",
            subject_normalized="invoice please",
            sender="user@example.com",
            recipients=["desk@co.com"],
            body_text="Body.",
            attachments=[],
            raw={},
        )
        msg.auth_spf = "not_run"
        msg.auth_dkim = "not_run"
        msg.auth_dmarc = "not_run"
        decision = triage(msg, rule_pack=_RULE_PACK)
        assert decision["auth_spf"] == "not_run"
        assert decision["auth_dkim"] == "not_run"
        assert decision["auth_dmarc"] == "not_run"
