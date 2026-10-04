# Phase 8 Release Notes

**Product:** Buro Assistant
**Release tag:** phase-8
**Date:** 2026-10-03
**Checklist rows closed:** 25, 26, 27, 28, 29, 30

---

## Summary

Phase 8 delivers the worker infrastructure: a priority-lane work queue with backpressure, a per-tenant daily token quota, a dead-letter queue with replay, cost and token fields on model decisions, and a lightweight trace log around every pipeline stage. A benchmark document is written to `docs/BENCHMARK.md`. No external message broker is required; everything is DB-backed and runs in the same process.

---

## What was built

### `app/workers/cost.py` — cost and token fields (row 25)

- `CostRecord(model, tokens_in, tokens_out)` — holds cost metadata for one model call.
- `compute_cost(model, tokens_in, tokens_out) → float` — estimates USD cost from a built-in per-model price table.
- `null_cost()` — returns a zero-cost record for rule-hit and fake-model paths.
- Migration `0005` adds `tokens_in`, `tokens_out`, `cost_usd` columns to the `decisions` table.

### `app/workers/quota.py` — per-tenant daily quota (row 26)

- `TenantQuota` ORM model (`quotas` table) — one row per `(tenant_id, quota_date)`.
- `check_quota(db, tenant_id, tokens_requested, policy_config)` — raises `QuotaExceeded` when `tokens_used + tokens_requested > daily_token_quota`. Default cap: 100 000 tokens/day.
- `record_usage(db, tenant_id, tokens_in, tokens_out, cost_usd)` — accumulates usage.
- `remaining_tokens(db, tenant_id, policy_config)` — returns tokens left today.
- Quota is strictly per-tenant; one tenant exhausting their budget does not affect another.

### `app/workers/queue.py` — priority lanes and backpressure (rows 27, 29)

- `WorkItem` ORM model (`work_queue` table) — state machine: `pending → processing → done / failed`.
- **Priority lanes (row 29):** `critical=0`, `high=1`, `medium=2`, `low=3`. `dequeue_next` always returns the smallest `lane` value first.
- **Backpressure (row 27):** `enqueue_work` raises `BackpressureError` when `queue_depth(tenant) >= queue_depth_cap` (default 500, configurable via tenant policy). Backpressure is per-tenant.
- `complete_item`, `fail_item`, `queue_depth` helpers.

### `app/workers/dlq.py` — dead-letter queue and replay (row 28)

- `DeadLetterItem` ORM model (`dead_letter` table).
- `send_to_dlq(db, tenant_id, …)` — moves a failed item to the DLQ.
- `replay(db, dlq_item_id, tenant_id, policy_config)` — re-enqueues the item with its original priority; marks `replayed_at` and increments `replay_count`. Cross-tenant replay returns `None`.
- `pending_dlq(db, tenant_id)` — returns unreplayed items newest-first.

### `app/workers/trace.py` — traces (row 30)

- `Trace` ORM model (`traces` table) — one row per pipeline stage per message.
- Stages: `ingest`, `decide`, `draft`, `send`.
- `record_trace(db, tenant_id, stage, started_at, …)` — writes one row; swallows DB errors silently so a trace failure never breaks the pipeline.
- `traced(db, tenant_id, stage)` — context manager that times the block, writes `status=ok` on success and `status=error` with `detail=str(exc)` on failure, then re-raises.

---

## Migration

`migrations/versions/0005_workers.py` — idempotent upgrade:
- Adds `tokens_in`, `tokens_out`, `cost_usd` to `decisions` (via `op.add_column` with existence check).
- Creates `traces`, `quotas`, `work_queue`, `dead_letter` tables.

---

## Tests

`tests/test_workers.py` — **40 tests** across 7 test classes:

| Class | Tests | Row |
|---|---|---|
| `TestCostFields` | 6 | 25 |
| `TestQuota` | 7 | 26 |
| `TestPriorityLanes` | 8 | 29 |
| `TestBackpressure` | 3 | 27 |
| `TestDLQ` | 6 | 28 |
| `TestTraces` | 7 | 30 |
| `TestBenchmark` | 2 | — (timing harness for BENCHMARK.md) |

Full suite: **279 tests, 0 failures**.

---

## Benchmark

`docs/BENCHMARK.md` — measured against in-memory SQLite on the development machine. Contains queue throughput, priority ordering, quota enforcement, DLQ replay latency, and trace overhead notes. No unmeasured throughput claim is printed.

---

## Capability matrix delta

| Row | Capability | Status |
|---|---|---|
| 25 | Cost and token fields on model calls | ✓ Phase 8 |
| 26 | Per-tenant daily quota | ✓ Phase 8 |
| 27 | Backpressure when queue depth exceeds tenant cap | ✓ Phase 8 |
| 28 | Dead-letter queue and replay | ✓ Phase 8 |
| 29 | Priority lanes | ✓ Phase 8 |
| 30 | Traces around ingest, decide, draft, and send | ✓ Phase 8 |

---

## What is not in this phase

- No live polling loop running in a background thread (by design — the worker is driven by Phase 9 / Phase 10 orchestration).
- No Phase 9 commercial controls (usage events, API keys, webhooks, retention).
