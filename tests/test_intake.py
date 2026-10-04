"""
tests/test_intake.py – intake worker loop tests (follow-up Phase 2).

Uses the FakeProvider so no real mailbox or IMAP connection is required.

Covered requirements:
  - Worker loop processes messages from the fake adapter
  - Failed pipeline jobs go to the dead-letter queue
  - Loop returns TickStats with correct counts
  - Processed messages reach an outcome (not left pending)
  - Failed jobs are DLQ'd, not silently dropped
  - Worker loop handles empty provider gracefully
  - Provider fetch error does not crash the loop
"""
from __future__ import annotations

import uuid
import pytest

from app.main import Base, engine, SessionLocal, Tenant, limiter
from app.ingest.normalize import NormalizedMessage
from app.ingest.providers.fake_provider import FakeProvider
from app.workers.intake_loop import run_once, TickStats
from app.workers.dlq import DeadLetterItem, pending_dlq


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


def _make_msg(tenant_id: int, subject: str = "Hello") -> NormalizedMessage:
    return NormalizedMessage(
        provider="fake",
        provider_message_id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        message_id_header=None,
        subject=subject,
        subject_normalized=NormalizedMessage.normalize_subject(subject),
        sender="sender@example.com",
        recipients=["inbox@example.com"],
        body_text="Please acknowledge receipt.",
        attachments=[],
        raw={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIntakeLoop:
    def test_empty_provider_returns_zero_stats(self):
        """Empty provider: no messages to process."""
        tenant = _make_tenant("il-empty")
        db = SessionLocal()
        try:
            provider = FakeProvider()  # no messages queued
            stats = run_once(db, tenant_id=tenant.id, provider=provider)
            assert stats.processed == 0
            assert stats.failed == 0
            assert stats.dead_lettered == 0
        finally:
            db.close()

    def test_single_message_processed(self):
        """One message in the fake provider → one processed."""
        tenant = _make_tenant("il-single")
        db = SessionLocal()
        try:
            provider = FakeProvider()
            provider.queue(_make_msg(tenant.id))
            stats = run_once(db, tenant_id=tenant.id, provider=provider)
            assert stats.processed == 1
            assert stats.failed == 0
        finally:
            db.close()

    def test_multiple_messages_all_processed(self):
        """Three messages → three processed, outcomes list has three entries."""
        tenant = _make_tenant("il-multi")
        db = SessionLocal()
        try:
            provider = FakeProvider()
            for i in range(3):
                provider.queue(_make_msg(tenant.id, subject=f"Subject {i}"))
            stats = run_once(db, tenant_id=tenant.id, provider=provider)
            assert stats.processed == 3
            assert len(stats.outcomes) == 3
            assert stats.failed == 0
        finally:
            db.close()

    def test_duplicate_message_counted_as_processed(self):
        """Duplicate is a valid pipeline outcome, not a failure."""
        tenant = _make_tenant("il-dup")
        db = SessionLocal()
        try:
            provider = FakeProvider()
            msg = _make_msg(tenant.id, subject="Same subject")
            # Queue the same provider_message_id twice would trigger idempotency;
            # instead send same subject twice to exercise duplicate detection.
            msg2 = NormalizedMessage(
                provider="fake",
                provider_message_id=msg.provider_message_id,   # same id → idempotency
                tenant_id=tenant.id,
                message_id_header=None,
                subject=msg.subject,
                subject_normalized=msg.subject_normalized,
                sender=msg.sender,
                recipients=msg.recipients,
                body_text=msg.body_text,
                attachments=[],
                raw={},
            )
            provider.queue(msg)
            provider.queue(msg2)
            stats = run_once(db, tenant_id=tenant.id, provider=provider)
            assert stats.processed == 2
            assert "duplicate" in stats.outcomes
            assert stats.failed == 0
        finally:
            db.close()

    def test_pipeline_exception_goes_to_dlq(self):
        """If the pipeline raises, the job is sent to the dead-letter queue."""
        tenant = _make_tenant("il-dlq")
        db = SessionLocal()

        # Build a provider that yields a bad NormalizedMessage to cause a
        # pipeline error by patching run_pipeline for this test only.
        provider = FakeProvider()
        provider.queue(_make_msg(tenant.id))

        # Monkey-patch run_pipeline to always raise for this test.
        import app.workers.intake_loop as _loop_module
        original = _loop_module.run_pipeline

        def _always_fail(db, msg, **kwargs):
            raise RuntimeError("simulated pipeline crash")

        _loop_module.run_pipeline = _always_fail
        try:
            stats = run_once(db, tenant_id=tenant.id, provider=provider)
            assert stats.processed == 0
            assert stats.failed == 1
            assert stats.dead_lettered == 1
            # Verify DLQ row was written.
            dlq_items = pending_dlq(db, tenant.id)
            assert len(dlq_items) == 1
            assert "simulated pipeline crash" in dlq_items[0].failure_reason
        finally:
            _loop_module.run_pipeline = original
            db.close()

    def test_provider_fetch_error_returns_empty_stats(self):
        """Provider.fetch_new raising must not crash the loop."""
        tenant = _make_tenant("il-fetch-err")

        class _BrokenProvider(FakeProvider):
            def fetch_new(self, tenant_id):
                raise ConnectionError("IMAP gone")

        db = SessionLocal()
        try:
            stats = run_once(db, tenant_id=tenant.id, provider=_BrokenProvider())
            assert stats.processed == 0
            assert stats.failed == 0
        finally:
            db.close()

    def test_stats_outcomes_match_processed_count(self):
        """len(outcomes) == processed for a clean run."""
        tenant = _make_tenant("il-outcomes")
        db = SessionLocal()
        try:
            provider = FakeProvider()
            for i in range(5):
                provider.queue(_make_msg(tenant.id, subject=f"Distinct {i}"))
            stats = run_once(db, tenant_id=tenant.id, provider=provider)
            assert len(stats.outcomes) == stats.processed
        finally:
            db.close()

    def test_fake_provider_is_used_when_no_imap_host(self, monkeypatch):
        """_make_provider returns FakeProvider when IMAP_HOST is absent."""
        monkeypatch.delenv("IMAP_HOST", raising=False)
        from app.workers.intake_loop import _make_provider
        from app.ingest.providers.fake_provider import FakeProvider as FP
        p = _make_provider()
        assert isinstance(p, FP)

    def test_imap_provider_selected_when_imap_host_set(self, monkeypatch):
        """_make_provider returns IMAPProvider when IMAP_HOST is present."""
        monkeypatch.setenv("IMAP_HOST", "imap.example.com")
        from app.workers.intake_loop import _make_provider
        from app.ingest.providers.imap_provider import IMAPProvider
        p = _make_provider()
        assert isinstance(p, IMAPProvider)
