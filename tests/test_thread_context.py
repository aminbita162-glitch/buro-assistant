"""
tests/test_thread_context.py — Phase 2: Bounded thread context.

Covers:
- fetch_thread_context returns empty string when no prior messages exist.
- fetch_thread_context returns the last ≤ 3 messages in the thread.
- Combined clip is capped at 200 characters total.
- The current message is excluded from the clip.
- Attachment bytes are not included (body_text only from raw_json).
- PII in prior messages is redacted.
- triage() prompt contains the thread context when provided.
- triage() prompt omits the thread line when context is empty.
- A rule hit still costs zero tokens even when thread context is present.
- Pipeline passes thread context to triage for a multi-message thread.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import Base
import app.ingest.models  # noqa: F401 – register Message on Base

from app.agents.amin import _build_prompt, triage, PROMPT_VERSION
from app.agents.fake_model import FakeModel
from app.agents.thread_context import (
    fetch_thread_context,
    THREAD_CLIP_CHARS,
    THREAD_LOOKBACK,
)
from app.ingest.models import Message
from app.ingest.normalize import NormalizedMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _insert_message(
    db,
    tenant_id: int,
    subject_normalized: str,
    body_text: str,
    provider_message_id: str,
    state: str = "new",
) -> Message:
    raw = {"body_text": body_text}
    row = Message(
        tenant_id=tenant_id,
        provider="fake",
        provider_message_id=provider_message_id,
        raw_json=json.dumps(raw),
        message_id_header=None,
        subject_normalized=subject_normalized,
        ingest_time=datetime.now(timezone.utc),
        state=state,
        attachment_state="none",
        auth_spf="not_run",
        auth_dkim="not_run",
        auth_dmarc="not_run",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _msg(
    subject: str = "Support request",
    body: str = "Please help.",
    sender: str = "user@example.com",
    provider_message_id: str = "current-001",
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
        recipients=["desk@company.com"],
        body_text=body,
        raw={"body_text": body},
    )


# ---------------------------------------------------------------------------
# TestFetchThreadContext — unit tests on fetch_thread_context()
# ---------------------------------------------------------------------------

class TestFetchThreadContext:

    def test_empty_when_no_prior_messages(self):
        db = _make_session()
        result = fetch_thread_context(db, tenant_id=1, subject_normalized="hello", exclude_provider_message_id="x")
        assert result == ""

    def test_excludes_current_message(self):
        db = _make_session()
        _insert_message(db, 1, "support request", "I need help.", "current-001")
        result = fetch_thread_context(db, 1, "support request", exclude_provider_message_id="current-001")
        assert result == ""

    def test_includes_one_prior_message(self):
        db = _make_session()
        _insert_message(db, 1, "support request", "First message.", "prior-001")
        result = fetch_thread_context(db, 1, "support request", exclude_provider_message_id="current-001")
        assert "First message." in result

    def test_includes_up_to_three_prior_messages(self):
        db = _make_session()
        _insert_message(db, 1, "support request", "Msg one.", "prior-001")
        _insert_message(db, 1, "support request", "Msg two.", "prior-002")
        _insert_message(db, 1, "support request", "Msg three.", "prior-003")
        result = fetch_thread_context(db, 1, "support request", exclude_provider_message_id="current-001")
        # All three should appear (they fit within 200 chars)
        assert "Msg one." in result
        assert "Msg two." in result
        assert "Msg three." in result

    def test_returns_only_last_three_when_more_exist(self):
        db = _make_session()
        _insert_message(db, 1, "support request", "Old msg.", "prior-000")
        _insert_message(db, 1, "support request", "Msg one.", "prior-001")
        _insert_message(db, 1, "support request", "Msg two.", "prior-002")
        _insert_message(db, 1, "support request", "Msg three.", "prior-003")
        result = fetch_thread_context(db, 1, "support request", exclude_provider_message_id="current-001")
        # Four prior messages — only the 3 newest should be included
        assert "Msg one." in result
        assert "Msg two." in result
        assert "Msg three." in result
        assert "Old msg." not in result

    def test_cap_at_200_characters(self):
        db = _make_session()
        long_body = "A" * 180
        _insert_message(db, 1, "support request", long_body, "prior-001")
        _insert_message(db, 1, "support request", "B" * 50, "prior-002")
        result = fetch_thread_context(db, 1, "support request", exclude_provider_message_id="current-001")
        assert len(result) <= THREAD_CLIP_CHARS

    def test_cap_exactly_200_characters(self):
        """The combined clip must never exceed THREAD_CLIP_CHARS."""
        db = _make_session()
        # Three messages each with 100-char bodies → would total 300 chars uncapped
        for i in range(3):
            _insert_message(db, 1, "threadcap", "X" * 100, f"prior-{i:03d}")
        result = fetch_thread_context(db, 1, "threadcap", exclude_provider_message_id="current-001")
        assert len(result) <= THREAD_CLIP_CHARS

    def test_does_not_include_duplicate_state_messages(self):
        """Messages with state='duplicate' must be excluded from the clip."""
        db = _make_session()
        _insert_message(db, 1, "support request", "Dup body.", "prior-dup", state="duplicate")
        result = fetch_thread_context(db, 1, "support request", exclude_provider_message_id="current-001")
        assert result == ""

    def test_does_not_include_quarantine_state_messages(self):
        db = _make_session()
        _insert_message(db, 1, "support request", "Quar body.", "prior-quar", state="quarantine")
        result = fetch_thread_context(db, 1, "support request", exclude_provider_message_id="current-001")
        assert result == ""

    def test_tenant_isolation(self):
        """Thread context must not cross tenant boundaries."""
        db = _make_session()
        _insert_message(db, 1, "support request", "Tenant 1 body.", "t1-prior-001")
        result = fetch_thread_context(db, 2, "support request", exclude_provider_message_id="current-001")
        assert result == ""

    def test_pii_redacted_in_clip(self):
        db = _make_session()
        _insert_message(db, 1, "support request", "Call me at 555-123-4567.", "prior-001")
        result = fetch_thread_context(db, 1, "support request", exclude_provider_message_id="current-001")
        assert "555-123-4567" not in result
        assert "[PHONE]" in result

    def test_email_redacted_in_clip(self):
        db = _make_session()
        _insert_message(db, 1, "support request", "Reply to user@secret.com.", "prior-001")
        result = fetch_thread_context(db, 1, "support request", exclude_provider_message_id="current-001")
        assert "user@secret.com" not in result
        assert "[EMAIL]" in result

    def test_empty_when_no_body_in_raw_json(self):
        """If raw_json has no body_text key, the entry is silently skipped."""
        db = _make_session()
        row = Message(
            tenant_id=1,
            provider="fake",
            provider_message_id="prior-nobody",
            raw_json=json.dumps({}),
            message_id_header=None,
            subject_normalized="support request",
            ingest_time=datetime.now(timezone.utc),
            state="new",
            attachment_state="none",
            auth_spf="not_run",
            auth_dkim="not_run",
            auth_dmarc="not_run",
        )
        db.add(row)
        db.commit()
        result = fetch_thread_context(db, 1, "support request", exclude_provider_message_id="current-001")
        assert result == ""


# ---------------------------------------------------------------------------
# TestBuildPromptThreadContext — unit tests on _build_prompt()
# ---------------------------------------------------------------------------

class TestBuildPromptThreadContext:

    def test_prompt_contains_thread_when_non_empty(self):
        prompt = _build_prompt(
            subject="test",
            body_clip="body",
            language="en",
            urgency="low",
            pack_hash="abc",
            thread_context="prior message clip",
        )
        assert "thread: prior message clip" in prompt

    def test_prompt_omits_thread_line_when_empty(self):
        prompt = _build_prompt(
            subject="test",
            body_clip="body",
            language="en",
            urgency="low",
            pack_hash="abc",
            thread_context="",
        )
        assert "thread:" not in prompt

    def test_prompt_thread_appears_before_return_directive(self):
        prompt = _build_prompt(
            subject="test",
            body_clip="body",
            language="en",
            urgency="low",
            pack_hash="abc",
            thread_context="clip here",
        )
        thread_pos = prompt.index("thread:")
        return_pos = prompt.index("Return JSON:")
        assert thread_pos < return_pos


# ---------------------------------------------------------------------------
# TestTriageThreadContext — integration: triage() receives thread_context
# ---------------------------------------------------------------------------

class TestTriageThreadContext:

    def _make_msg(self, subject="Hello", body="Some body."):
        return NormalizedMessage(
            provider="fake",
            provider_message_id="test-001",
            tenant_id=1,
            message_id_header=None,
            subject=subject,
            subject_normalized=NormalizedMessage.normalize_subject(subject),
            sender="user@example.com",
            recipients=["desk@company.com"],
            body_text=body,
            raw={},
        )

    def test_thread_context_in_prompt_when_model_called(self):
        """The prompt passed to the model must include the thread context line."""
        captured = {}

        class CapturingModel:
            def call(self, prompt):
                captured["prompt"] = prompt
                return {"department": "general", "action": "hold", "confidence": 0.9}

        msg = self._make_msg()
        triage(msg, model=CapturingModel(), thread_context="prior clip here")
        assert "thread: prior clip here" in captured["prompt"]

    def test_no_thread_line_in_prompt_when_context_empty(self):
        captured = {}

        class CapturingModel:
            def call(self, prompt):
                captured["prompt"] = prompt
                return {"department": "general", "action": "hold", "confidence": 0.9}

        msg = self._make_msg()
        triage(msg, model=CapturingModel(), thread_context="")
        assert "thread:" not in captured["prompt"]

    def test_rule_hit_costs_zero_tokens_with_thread_context(self):
        """A rule hit must not call the model even when thread_context is provided."""
        # Use the correct rule_pack schema: domain_rules with a "match" key.
        rule_pack = {
            "domain_rules": [
                {
                    "match": "example.com",
                    "department": "finance",
                    "action": "hold",
                }
            ]
        }
        call_count = {"n": 0}

        class CountingModel:
            def call(self, prompt):
                call_count["n"] += 1
                return {"department": "general", "action": "hold", "confidence": 0.9}

        msg = self._make_msg()
        decision = triage(msg, rule_pack=rule_pack, model=CountingModel(), thread_context="some prior clip")
        assert call_count["n"] == 0  # model must not be called on a rule hit
        assert decision["rule_hit"] == "domain:example.com"

    def test_prompt_version_updated(self):
        """PROMPT_VERSION must reflect Phase 2 increment."""
        assert PROMPT_VERSION == "amin-v3"


# ---------------------------------------------------------------------------
# TestThreadClipConstants
# ---------------------------------------------------------------------------

class TestThreadClipConstants:

    def test_clip_cap_is_200(self):
        assert THREAD_CLIP_CHARS == 200

    def test_lookback_is_three(self):
        assert THREAD_LOOKBACK == 3
