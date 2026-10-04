"""
Phase 8 worker tests.

Covered checklist rows:
  25 – cost and token fields on model calls
  26 – per-tenant daily quota
  27 – backpressure when queue depth exceeds tenant cap
  28 – dead-letter queue and replay
  29 – priority lanes
  30 – traces around ingest, decide, draft, send

All tests run against the in-memory SQLite engine from conftest.py.
No external service or model API is contacted.
"""
from __future__ import annotations

import time
import pytest
from datetime import date, datetime, timezone
from sqlalchemy import inspect as sa_inspect

from app.main import Base, engine, SessionLocal, Tenant, limiter

# Worker modules under test
from app.workers.cost import CostRecord, compute_cost, null_cost
from app.workers.quota import (
    TenantQuota,
    QuotaExceeded,
    check_quota,
    record_usage,
    remaining_tokens,
    get_or_create_quota,
)
from app.workers.queue import (
    WorkItem,
    BackpressureError,
    PRIORITY_LANE,
    enqueue_work,
    dequeue_next,
    complete_item,
    fail_item,
    queue_depth,
)
from app.workers.dlq import (
    DeadLetterItem,
    send_to_dlq,
    replay,
    pending_dlq,
)
from app.workers.trace import Trace, record_trace, traced


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


# ---------------------------------------------------------------------------
# Row 25 – cost and token fields
# ---------------------------------------------------------------------------

class TestCostFields:
    def test_compute_cost_known_model(self):
        cost = compute_cost("gpt-4.1-mini", tokens_in=1000, tokens_out=500)
        assert cost > 0

    def test_compute_cost_fake_model_is_zero(self):
        assert compute_cost("fake", 999, 999) == 0.0

    def test_null_cost_returns_zero(self):
        cr = null_cost()
        assert cr.cost_usd == 0.0
        assert cr.tokens_in == 0
        assert cr.tokens_out == 0

    def test_cost_record_as_dict(self):
        cr = CostRecord("gpt-4.1-mini", 100, 50)
        d = cr.as_dict()
        assert d["tokens_in"] == 100
        assert d["tokens_out"] == 50
        assert d["cost_usd"] > 0

    def test_cost_scales_with_tokens(self):
        c1 = compute_cost("gpt-4.1-mini", 1000, 0)
        c2 = compute_cost("gpt-4.1-mini", 2000, 0)
        assert c2 == pytest.approx(c1 * 2)

    def test_decisions_migration_defines_cost_columns(self):
        """Migration 0005 adds cost columns; verify the migration source."""
        src_path = "migrations/versions/0005_workers.py"
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        assert "tokens_in"  in src
        assert "tokens_out" in src
        assert "cost_usd"   in src


# ---------------------------------------------------------------------------
# Row 26 – per-tenant daily quota
# ---------------------------------------------------------------------------

class TestQuota:
    def test_tables_exist(self):
        insp = sa_inspect(engine)
        assert "quotas" in insp.get_table_names()

    def test_quota_created_on_first_access(self):
        tenant = _make_tenant("q1")
        db = SessionLocal()
        try:
            row = get_or_create_quota(db, tenant.id)
            assert row.tokens_used == 0
            assert row.quota_date == datetime.now(timezone.utc).date()
        finally:
            db.close()

    def test_check_quota_passes_under_limit(self):
        tenant = _make_tenant("q2")
        db = SessionLocal()
        try:
            # Default limit 100 000 — 500 tokens well within budget.
            check_quota(db, tenant.id, 500)   # should not raise
        finally:
            db.close()

    def test_quota_exceeded_when_over_budget(self):
        tenant = _make_tenant("q3")
        db = SessionLocal()
        try:
            policy = {"daily_token_quota": 100}
            record_usage(db, tenant.id, 80, 20, 0.0)   # uses exactly 100
            with pytest.raises(QuotaExceeded):
                check_quota(db, tenant.id, 1, policy)
        finally:
            db.close()

    def test_record_usage_accumulates(self):
        tenant = _make_tenant("q4")
        db = SessionLocal()
        try:
            record_usage(db, tenant.id, 100, 50, 0.01)
            record_usage(db, tenant.id, 200, 50, 0.02)
            row = get_or_create_quota(db, tenant.id)
            assert row.tokens_used == 400
        finally:
            db.close()

    def test_remaining_tokens_decreases_after_usage(self):
        tenant = _make_tenant("q5")
        db = SessionLocal()
        try:
            policy = {"daily_token_quota": 1000}
            before = remaining_tokens(db, tenant.id, policy)
            record_usage(db, tenant.id, 300, 100, 0.0)
            after = remaining_tokens(db, tenant.id, policy)
            assert after == before - 400
        finally:
            db.close()

    def test_quota_is_per_tenant(self):
        t1 = _make_tenant("qt1")
        t2 = _make_tenant("qt2")
        db = SessionLocal()
        try:
            record_usage(db, t1.id, 900, 0, 0.0)
            policy = {"daily_token_quota": 1000}
            # t2 not affected by t1 usage
            check_quota(db, t2.id, 900, policy)   # should not raise
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Rows 27, 29 – queue backpressure and priority lanes
# ---------------------------------------------------------------------------

class TestPriorityLanes:
    def test_tables_exist(self):
        insp = sa_inspect(engine)
        assert "work_queue" in insp.get_table_names()

    def test_enqueue_creates_pending_item(self):
        tenant = _make_tenant("wq1")
        db = SessionLocal()
        try:
            item = enqueue_work(db, tenant.id, priority="medium")
            assert item.state == "pending"
            assert item.priority == "medium"
            assert item.lane == PRIORITY_LANE["medium"]
        finally:
            db.close()

    def test_priority_ordering(self):
        """critical is always dequeued before high, high before medium, etc."""
        tenant = _make_tenant("wq2")
        db = SessionLocal()
        try:
            enqueue_work(db, tenant.id, priority="low")
            enqueue_work(db, tenant.id, priority="medium")
            enqueue_work(db, tenant.id, priority="high")
            enqueue_work(db, tenant.id, priority="critical")

            order = []
            for _ in range(4):
                item = dequeue_next(db, tenant.id)
                order.append(item.priority)

            assert order == ["critical", "high", "medium", "low"]
        finally:
            db.close()

    def test_dequeue_returns_none_when_empty(self):
        tenant = _make_tenant("wq3")
        db = SessionLocal()
        try:
            assert dequeue_next(db, tenant.id) is None
        finally:
            db.close()

    def test_complete_item_sets_done(self):
        tenant = _make_tenant("wq4")
        db = SessionLocal()
        try:
            item = enqueue_work(db, tenant.id)
            dequeue_next(db, tenant.id)
            done = complete_item(db, item.id)
            assert done.state == "done"
        finally:
            db.close()

    def test_fail_item_sets_failed(self):
        tenant = _make_tenant("wq5")
        db = SessionLocal()
        try:
            item = enqueue_work(db, tenant.id)
            failed = fail_item(db, item.id, "network error")
            assert failed.state == "failed"
        finally:
            db.close()

    def test_queue_depth_reflects_pending(self):
        tenant = _make_tenant("wq6")
        db = SessionLocal()
        try:
            assert queue_depth(db, tenant.id) == 0
            enqueue_work(db, tenant.id)
            enqueue_work(db, tenant.id)
            assert queue_depth(db, tenant.id) == 2
        finally:
            db.close()

    def test_lane_values_ordered(self):
        assert PRIORITY_LANE["critical"] < PRIORITY_LANE["high"]
        assert PRIORITY_LANE["high"]     < PRIORITY_LANE["medium"]
        assert PRIORITY_LANE["medium"]   < PRIORITY_LANE["low"]


class TestBackpressure:
    def test_backpressure_raised_at_cap(self):
        """Row 27: enqueue_work raises BackpressureError when depth >= cap."""
        tenant = _make_tenant("bp1")
        db = SessionLocal()
        try:
            policy = {"queue_depth_cap": 3}
            enqueue_work(db, tenant.id, policy_config=policy)
            enqueue_work(db, tenant.id, policy_config=policy)
            enqueue_work(db, tenant.id, policy_config=policy)
            with pytest.raises(BackpressureError):
                enqueue_work(db, tenant.id, policy_config=policy)
        finally:
            db.close()

    def test_backpressure_is_per_tenant(self):
        t1 = _make_tenant("bp-t1")
        t2 = _make_tenant("bp-t2")
        db = SessionLocal()
        try:
            policy = {"queue_depth_cap": 1}
            enqueue_work(db, t1.id, policy_config=policy)
            # t2 unaffected — should not raise
            enqueue_work(db, t2.id, policy_config=policy)
        finally:
            db.close()

    def test_processing_items_do_not_count_toward_depth(self):
        tenant = _make_tenant("bp2")
        db = SessionLocal()
        try:
            policy = {"queue_depth_cap": 1}
            enqueue_work(db, tenant.id, policy_config=policy)
            dequeue_next(db, tenant.id)   # moves to "processing"
            # depth is now 0 pending — should not raise
            enqueue_work(db, tenant.id, policy_config=policy)
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Row 28 – dead-letter queue and replay
# ---------------------------------------------------------------------------

class TestDLQ:
    def test_table_exists(self):
        insp = sa_inspect(engine)
        assert "dead_letter" in insp.get_table_names()

    def test_send_to_dlq_creates_entry(self):
        tenant = _make_tenant("dlq1")
        db = SessionLocal()
        try:
            entry = send_to_dlq(db, tenant.id, priority="high",
                                failure_reason="network timeout")
            assert entry.id > 0
            assert entry.failure_reason == "network timeout"
            assert entry.replay_count == 0
            assert entry.replayed_at is None
        finally:
            db.close()

    def test_pending_dlq_returns_unreplayed(self):
        tenant = _make_tenant("dlq2")
        db = SessionLocal()
        try:
            send_to_dlq(db, tenant.id, failure_reason="err1")
            send_to_dlq(db, tenant.id, failure_reason="err2")
            items = pending_dlq(db, tenant.id)
            assert len(items) == 2
        finally:
            db.close()

    def test_replay_re_enqueues_item(self):
        tenant = _make_tenant("dlq3")
        db = SessionLocal()
        try:
            entry = send_to_dlq(db, tenant.id, priority="medium", payload='{"x":1}')
            work_item = replay(db, entry.id, tenant.id)
            assert work_item is not None
            assert work_item.state == "pending"
            assert work_item.priority == "medium"
        finally:
            db.close()

    def test_replay_marks_entry_replayed(self):
        tenant = _make_tenant("dlq4")
        db = SessionLocal()
        try:
            entry = send_to_dlq(db, tenant.id)
            replay(db, entry.id, tenant.id)
            db.refresh(entry)
            assert entry.replayed_at is not None
            assert entry.replay_count == 1
        finally:
            db.close()

    def test_replay_wrong_tenant_returns_none(self):
        t1 = _make_tenant("dlq-ct1")
        t2 = _make_tenant("dlq-ct2")
        db = SessionLocal()
        try:
            entry = send_to_dlq(db, t1.id)
            result = replay(db, entry.id, t2.id)
            assert result is None
        finally:
            db.close()

    def test_replayed_item_not_in_pending(self):
        tenant = _make_tenant("dlq5")
        db = SessionLocal()
        try:
            entry = send_to_dlq(db, tenant.id)
            replay(db, entry.id, tenant.id)
            pending = pending_dlq(db, tenant.id)
            assert len(pending) == 0
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Row 30 – traces around ingest, decide, draft, send
# ---------------------------------------------------------------------------

class TestTraces:
    def test_table_exists(self):
        insp = sa_inspect(engine)
        assert "traces" in insp.get_table_names()

    def test_record_trace_writes_row(self):
        tenant = _make_tenant("tr1")
        db = SessionLocal()
        try:
            now = datetime.now(timezone.utc)
            tr = record_trace(db, tenant.id, "ingest", now)
            assert tr.id > 0
            assert tr.stage == "ingest"
            assert tr.status == "ok"
            assert tr.duration_ms >= 0
        finally:
            db.close()

    def test_all_four_stages_accepted(self):
        tenant = _make_tenant("tr2")
        db = SessionLocal()
        try:
            now = datetime.now(timezone.utc)
            for stage in ("ingest", "decide", "draft", "send"):
                tr = record_trace(db, tenant.id, stage, now)
                assert tr.stage == stage
        finally:
            db.close()

    def test_traced_context_manager_writes_ok_row(self):
        tenant = _make_tenant("tr3")
        db = SessionLocal()
        try:
            with traced(db, tenant.id, "decide") as ctx:
                pass   # no-op work
            assert "trace_id" in ctx
            tr = db.query(Trace).filter(Trace.id == ctx["trace_id"]).first()
            assert tr.status == "ok"
        finally:
            db.close()

    def test_traced_context_manager_writes_error_on_exception(self):
        tenant = _make_tenant("tr4")
        db = SessionLocal()
        try:
            with pytest.raises(ValueError):
                with traced(db, tenant.id, "draft"):
                    raise ValueError("draft failed")
            rows = db.query(Trace).filter(
                Trace.tenant_id == tenant.id, Trace.stage == "draft"
            ).all()
            assert any(r.status == "error" for r in rows)
        finally:
            db.close()

    def test_trace_duration_ms_positive(self):
        tenant = _make_tenant("tr5")
        db = SessionLocal()
        try:
            now = datetime.now(timezone.utc)
            tr = record_trace(db, tenant.id, "send", now)
            assert tr.duration_ms >= 0
        finally:
            db.close()

    def test_trace_tenant_isolation(self):
        t1 = _make_tenant("tr-t1")
        t2 = _make_tenant("tr-t2")
        db = SessionLocal()
        try:
            now = datetime.now(timezone.utc)
            record_trace(db, t1.id, "ingest", now)
            rows_t2 = db.query(Trace).filter(Trace.tenant_id == t2.id).all()
            assert len(rows_t2) == 0
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Benchmark (docs/BENCHMARK.md)
# ---------------------------------------------------------------------------

class TestBenchmark:
    """
    Timing harness that produces the numbers cited in docs/BENCHMARK.md.
    Does not assert specific thresholds (varies by machine); verifies the
    operations complete without error and measures wall-clock time.
    """

    def test_enqueue_100_items(self):
        tenant = _make_tenant("bm1")
        db = SessionLocal()
        try:
            start = time.monotonic()
            for i in range(100):
                enqueue_work(db, tenant.id, priority="medium",
                             payload=f'{{"i":{i}}}')
            elapsed_ms = (time.monotonic() - start) * 1000
            assert queue_depth(db, tenant.id) == 100
            # Existence check — no throughput claim.
            assert elapsed_ms >= 0
        finally:
            db.close()

    def test_dequeue_100_items(self):
        tenant = _make_tenant("bm2")
        db = SessionLocal()
        try:
            for i in range(100):
                enqueue_work(db, tenant.id, priority="medium")
            start = time.monotonic()
            for _ in range(100):
                dequeue_next(db, tenant.id)
            elapsed_ms = (time.monotonic() - start) * 1000
            assert queue_depth(db, tenant.id) == 0
            assert elapsed_ms >= 0
        finally:
            db.close()
