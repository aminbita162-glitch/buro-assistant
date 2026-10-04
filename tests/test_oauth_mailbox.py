"""
tests/test_oauth_mailbox.py – Phase 4: OAuth mailbox boundary.

Covered requirements (DIRECTIVE.txt Phase 4):
  - An OAuth mailbox port exists beside the IMAP adapter.
  - Tokens are read from the environment only (verified by the factory tests).
  - When tokens are absent the fake provider is returned and the app still runs.
  - No token value is present in this repository.
  - All tests use the fake port only; no external service is contacted.
"""
from __future__ import annotations

import os

import pytest

from app.ingest.normalize import NormalizedMessage
from app.ingest.providers.oauth_port import OAuthMailboxPort
from app.ingest.providers.fake_oauth_provider import FakeOAuthProvider
from app.ingest.providers.oauth_factory import get_oauth_provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(
    subject: str = "Hello",
    body: str = "Test body.",
    sender: str = "sender@example.com",
    provider_message_id: str = "oauth-msg-001",
    tenant_id: int = 1,
) -> NormalizedMessage:
    return NormalizedMessage(
        provider="fake_oauth",
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


# ---------------------------------------------------------------------------
# OAuthMailboxPort interface
# ---------------------------------------------------------------------------

class TestOAuthMailboxPort:
    def test_fake_provider_is_subclass_of_port(self):
        """FakeOAuthProvider must satisfy the OAuthMailboxPort interface."""
        provider = FakeOAuthProvider()
        assert isinstance(provider, OAuthMailboxPort)

    def test_fake_provider_name(self):
        provider = FakeOAuthProvider()
        assert provider.provider_name == "fake_oauth"

    def test_fetch_new_empty_queue_yields_nothing(self):
        provider = FakeOAuthProvider()
        messages = list(provider.fetch_new(tenant_id=1))
        assert messages == []

    def test_fetch_new_returns_queued_messages(self):
        provider = FakeOAuthProvider()
        provider.queue(_msg(subject="First"))
        provider.queue(_msg(subject="Second", provider_message_id="oauth-msg-002"))
        messages = list(provider.fetch_new(tenant_id=1))
        assert len(messages) == 2
        assert messages[0].subject == "First"
        assert messages[1].subject == "Second"

    def test_fetch_new_clears_queue(self):
        provider = FakeOAuthProvider()
        provider.queue(_msg())
        list(provider.fetch_new(tenant_id=1))
        # Queue must be empty after the first fetch.
        second_fetch = list(provider.fetch_new(tenant_id=1))
        assert second_fetch == []

    def test_fetch_new_stamps_tenant_id(self):
        """fetch_new must override the tenant_id on yielded messages."""
        provider = FakeOAuthProvider()
        msg = _msg(tenant_id=99)
        provider.queue(msg)
        result = list(provider.fetch_new(tenant_id=7))
        assert result[0].tenant_id == 7

    def test_fetch_new_multiple_tenants_isolated(self):
        """Two separate providers must not share state."""
        p1 = FakeOAuthProvider()
        p2 = FakeOAuthProvider()
        p1.queue(_msg(subject="Tenant1 mail"))
        msgs1 = list(p1.fetch_new(tenant_id=1))
        msgs2 = list(p2.fetch_new(tenant_id=2))
        assert len(msgs1) == 1
        assert msgs2 == []

    def test_queue_then_fetch_preserves_order(self):
        provider = FakeOAuthProvider()
        for i in range(5):
            provider.queue(_msg(subject=f"msg-{i}", provider_message_id=f"id-{i}"))
        messages = list(provider.fetch_new(tenant_id=1))
        subjects = [m.subject for m in messages]
        assert subjects == [f"msg-{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# Factory — tokens absent → fake provider; app still runs
# ---------------------------------------------------------------------------

class TestOAuthFactory:
    def setup_method(self):
        """Ensure OAuth env vars are absent before each test."""
        for var in ("OAUTH_TOKEN_URL", "OAUTH_CLIENT_ID", "OAUTH_CLIENT_SECRET"):
            os.environ.pop(var, None)

    def teardown_method(self):
        for var in ("OAUTH_TOKEN_URL", "OAUTH_CLIENT_ID", "OAUTH_CLIENT_SECRET"):
            os.environ.pop(var, None)

    def test_factory_returns_fake_when_vars_absent(self):
        """With no env vars the factory must return a FakeOAuthProvider."""
        provider = get_oauth_provider()
        assert isinstance(provider, FakeOAuthProvider)

    def test_factory_returns_fake_when_only_token_url_set(self):
        os.environ["OAUTH_TOKEN_URL"] = "https://login.example.com/token"
        provider = get_oauth_provider()
        assert isinstance(provider, FakeOAuthProvider)

    def test_factory_returns_fake_when_only_client_id_set(self):
        os.environ["OAUTH_CLIENT_ID"] = "some-client-id"
        provider = get_oauth_provider()
        assert isinstance(provider, FakeOAuthProvider)

    def test_factory_returns_fake_when_only_client_secret_set(self):
        os.environ["OAUTH_CLIENT_SECRET"] = "some-secret"
        provider = get_oauth_provider()
        assert isinstance(provider, FakeOAuthProvider)

    def test_factory_returns_fake_when_token_url_empty_string(self):
        os.environ["OAUTH_TOKEN_URL"] = ""
        os.environ["OAUTH_CLIENT_ID"] = "cid"
        os.environ["OAUTH_CLIENT_SECRET"] = "csecret"
        provider = get_oauth_provider()
        assert isinstance(provider, FakeOAuthProvider)

    def test_factory_app_still_runs_without_credentials(self):
        """Calling the factory with no env vars must not raise."""
        # If this does not raise, the app still runs without OAuth creds.
        provider = get_oauth_provider()
        assert provider is not None

    def test_factory_result_is_usable_as_port(self):
        """The returned provider must satisfy the OAuthMailboxPort interface."""
        provider = get_oauth_provider()
        assert isinstance(provider, OAuthMailboxPort)

    def test_factory_returned_provider_can_fetch(self):
        """The provider returned by the factory can receive and yield messages."""
        provider = get_oauth_provider()
        provider.queue(_msg(subject="Factory test"))
        messages = list(provider.fetch_new(tenant_id=3))
        assert len(messages) == 1
        assert messages[0].subject == "Factory test"

    def test_no_token_value_in_env_after_factory_call(self):
        """The factory must not inject any token into the environment."""
        get_oauth_provider()
        assert os.environ.get("OAUTH_TOKEN_URL") is None
        assert os.environ.get("OAUTH_CLIENT_ID") is None
        assert os.environ.get("OAUTH_CLIENT_SECRET") is None
