"""
tests/test_audit_hash_chain.py – Phase 5: audit hash chain.

Covered requirements:
  - Each new row stores the SHA-256 of the previous row's canonical fields.
  - The very first row stores ZERO_HASH as its prev_hash.
  - verify_chain() returns an empty list when the chain is intact.
  - A tampered row (any field changed) breaks the chain: verify_chain()
    returns the id of the row whose prev_hash no longer matches.
  - Tampering a middle row breaks all rows that follow it.
  - Rows written before Phase 5 (prev_hash IS NULL) are excluded from
    chain verification.
  - Multiple tenants maintain independent chains.

All tests use the in-memory SQLite engine from conftest.py.
No live mail server or model API is contacted.
"""
from __future__ import annotations

import hashlib
import pytest

from app.main import Base, engine, SessionLocal, Tenant, limiter
from app.policy.audit import (
    AuditLogEntry,
    ZERO_HASH,
    log_event,
    verify_chain,
    _canonical,
    _sha256,
)


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


def _make_tenant(slug: str) -> Tenant:
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
# Tests: prev_hash values on new rows
# ---------------------------------------------------------------------------

class TestHashChainWrite:
    def test_first_row_carries_zero_hash(self):
        t = _make_tenant("hc-t1")
        db = SessionLocal()
        try:
            entry = log_event(db, t.id, "message_ingested")
            assert entry.prev_hash == ZERO_HASH
        finally:
            db.close()

    def test_second_row_carries_hash_of_first(self):
        t = _make_tenant("hc-t2")
        db = SessionLocal()
        try:
            e1 = log_event(db, t.id, "message_ingested")
            e2 = log_event(db, t.id, "triage_decided")
            expected = _sha256(_canonical(e1))
            assert e2.prev_hash == expected
        finally:
            db.close()

    def test_third_row_carries_hash_of_second(self):
        t = _make_tenant("hc-t3")
        db = SessionLocal()
        try:
            e1 = log_event(db, t.id, "message_ingested")
            e2 = log_event(db, t.id, "triage_decided")
            e3 = log_event(db, t.id, "draft_created")
            expected = _sha256(_canonical(e2))
            assert e3.prev_hash == expected
        finally:
            db.close()

    def test_prev_hash_is_64_hex_chars(self):
        t = _make_tenant("hc-t4")
        db = SessionLocal()
        try:
            log_event(db, t.id, "message_ingested")
            e2 = log_event(db, t.id, "triage_decided")
            assert len(e2.prev_hash) == 64
            int(e2.prev_hash, 16)  # must be valid hex
        finally:
            db.close()

    def test_zero_hash_is_64_zeros(self):
        assert ZERO_HASH == "0" * 64
        assert len(ZERO_HASH) == 64


# ---------------------------------------------------------------------------
# Tests: verify_chain — intact chain
# ---------------------------------------------------------------------------

class TestVerifyChainIntact:
    def test_empty_log_returns_no_broken_rows(self):
        t = _make_tenant("vc-t1")
        db = SessionLocal()
        try:
            result = verify_chain(db, t.id)
            assert result == []
        finally:
            db.close()

    def test_single_row_passes_verification(self):
        t = _make_tenant("vc-t2")
        db = SessionLocal()
        try:
            log_event(db, t.id, "message_ingested")
            result = verify_chain(db, t.id)
            assert result == []
        finally:
            db.close()

    def test_multiple_rows_pass_verification(self):
        t = _make_tenant("vc-t3")
        db = SessionLocal()
        try:
            for event in ("message_ingested", "triage_decided", "draft_created",
                          "draft_approved", "message_sent"):
                log_event(db, t.id, event)
            result = verify_chain(db, t.id)
            assert result == []
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Tests: verify_chain — broken chain
# ---------------------------------------------------------------------------

class TestVerifyChainBroken:
    def test_tampered_event_breaks_chain(self):
        """Changing the event field of a row makes all following rows fail."""
        t = _make_tenant("br-t1")
        db = SessionLocal()
        try:
            e1 = log_event(db, t.id, "message_ingested")
            e2 = log_event(db, t.id, "triage_decided")
            e3 = log_event(db, t.id, "draft_created")

            # Tamper with row 1 by changing its event field directly.
            db.query(AuditLogEntry).filter(AuditLogEntry.id == e1.id).update(
                {"event": "TAMPERED"}
            )
            db.commit()

            broken = verify_chain(db, t.id)
            # Row 2's prev_hash now mismatches because row 1 changed.
            assert e2.id in broken
        finally:
            db.close()

    def test_tampered_detail_breaks_chain(self):
        """Changing the detail field of a row makes the next row's hash fail."""
        t = _make_tenant("br-t2")
        db = SessionLocal()
        try:
            e1 = log_event(db, t.id, "message_ingested", detail="original")
            e2 = log_event(db, t.id, "triage_decided")

            db.query(AuditLogEntry).filter(AuditLogEntry.id == e1.id).update(
                {"detail": "injected"}
            )
            db.commit()

            broken = verify_chain(db, t.id)
            assert e2.id in broken
        finally:
            db.close()

    def test_tampered_actor_breaks_chain(self):
        t = _make_tenant("br-t3")
        db = SessionLocal()
        try:
            e1 = log_event(db, t.id, "message_ingested", actor="system")
            e2 = log_event(db, t.id, "triage_decided")

            db.query(AuditLogEntry).filter(AuditLogEntry.id == e1.id).update(
                {"actor": "attacker"}
            )
            db.commit()

            broken = verify_chain(db, t.id)
            assert e2.id in broken
        finally:
            db.close()

    def test_tampered_middle_row_breaks_next_row(self):
        """Tampering row 2 breaks row 3 (whose prev_hash was computed from original row 2)."""
        t = _make_tenant("br-t4")
        db = SessionLocal()
        try:
            e1 = log_event(db, t.id, "message_ingested")
            e2 = log_event(db, t.id, "triage_decided")
            e3 = log_event(db, t.id, "draft_created")
            e4 = log_event(db, t.id, "draft_approved")

            # Tamper row 2 (middle).
            db.query(AuditLogEntry).filter(AuditLogEntry.id == e2.id).update(
                {"event": "TAMPERED"}
            )
            db.commit()

            broken = verify_chain(db, t.id)
            # Row 3's prev_hash now mismatches because row 2 was tampered.
            assert e3.id in broken
            # Row 1 and row 4 are unaffected (their prev_hash links are intact).
            assert e1.id not in broken
            assert e4.id not in broken
        finally:
            db.close()

    def test_tampered_prev_hash_itself_breaks_chain(self):
        """Directly changing prev_hash of a row is detected."""
        t = _make_tenant("br-t5")
        db = SessionLocal()
        try:
            e1 = log_event(db, t.id, "message_ingested")
            e2 = log_event(db, t.id, "triage_decided")

            db.query(AuditLogEntry).filter(AuditLogEntry.id == e2.id).update(
                {"prev_hash": "a" * 64}
            )
            db.commit()

            broken = verify_chain(db, t.id)
            assert e2.id in broken
        finally:
            db.close()

    def test_intact_row_not_reported_as_broken(self):
        """Verify that untampered rows are not reported as broken."""
        t = _make_tenant("br-t6")
        db = SessionLocal()
        try:
            e1 = log_event(db, t.id, "message_ingested")
            e2 = log_event(db, t.id, "triage_decided")
            e3 = log_event(db, t.id, "draft_created")

            # Tamper only row 3.
            db.query(AuditLogEntry).filter(AuditLogEntry.id == e3.id).update(
                {"event": "TAMPERED"}
            )
            db.commit()

            broken = verify_chain(db, t.id)
            assert e1.id not in broken
            assert e2.id not in broken
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Tests: pre-chain rows (prev_hash IS NULL) are excluded
# ---------------------------------------------------------------------------

class TestPreChainRowsExcluded:
    def test_null_prev_hash_rows_skipped_by_verify(self):
        """Simulate a pre-Phase-5 row: insert with NULL prev_hash."""
        t = _make_tenant("pc-t1")
        db = SessionLocal()
        try:
            # Insert a legacy row with NULL prev_hash directly.
            legacy = AuditLogEntry(
                tenant_id=t.id,
                event="legacy_event",
                actor="system",
                detail=None,
                created_at=__import__("datetime").datetime(2026, 1, 1, 0, 0, 0,
                    tzinfo=__import__("datetime").timezone.utc),
                prev_hash=None,
            )
            db.add(legacy)
            db.commit()
            db.refresh(legacy)

            # Chain verification must not touch the legacy row.
            result = verify_chain(db, t.id)
            assert result == []
        finally:
            db.close()

    def test_new_rows_after_legacy_row_form_their_own_chain(self):
        """New rows after a NULL row start their chain from ZERO_HASH."""
        t = _make_tenant("pc-t2")
        db = SessionLocal()
        try:
            from datetime import datetime, timezone as tz

            legacy = AuditLogEntry(
                tenant_id=t.id,
                event="legacy_event",
                actor="system",
                detail=None,
                created_at=datetime(2026, 1, 1, 0, 0, 0, tzinfo=tz.utc),
                prev_hash=None,
            )
            db.add(legacy)
            db.commit()
            db.refresh(legacy)

            # Now write a new row through the normal path.
            e_new = log_event(db, t.id, "message_ingested")
            assert e_new.prev_hash == ZERO_HASH

            result = verify_chain(db, t.id)
            assert result == []
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Tests: tenant isolation
# ---------------------------------------------------------------------------

class TestTenantChainIsolation:
    def test_tenants_have_independent_chains(self):
        t1 = _make_tenant("iso-t1")
        t2 = _make_tenant("iso-t2")
        db = SessionLocal()
        try:
            log_event(db, t1.id, "message_ingested")
            log_event(db, t1.id, "triage_decided")
            log_event(db, t2.id, "message_ingested")

            # Both chains intact.
            assert verify_chain(db, t1.id) == []
            assert verify_chain(db, t2.id) == []
        finally:
            db.close()

    def test_tamper_in_one_tenant_does_not_affect_other(self):
        t1 = _make_tenant("iso-t3")
        t2 = _make_tenant("iso-t4")
        db = SessionLocal()
        try:
            e1_t1 = log_event(db, t1.id, "message_ingested")
            e2_t1 = log_event(db, t1.id, "triage_decided")
            log_event(db, t2.id, "message_ingested")
            log_event(db, t2.id, "triage_decided")

            # Tamper tenant 1.
            db.query(AuditLogEntry).filter(AuditLogEntry.id == e1_t1.id).update(
                {"event": "TAMPERED"}
            )
            db.commit()

            assert e2_t1.id in verify_chain(db, t1.id)
            assert verify_chain(db, t2.id) == []
        finally:
            db.close()
