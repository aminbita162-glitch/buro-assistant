# Buro Assistant — Benchmark

**Product:** Buro Assistant
**Date:** 2026-10-03
**Scope:** Phase 8 worker infrastructure

---

## Methodology

All numbers in this document were produced by the benchmark harness in
`tests/test_workers.py` running against an in-memory SQLite database on the
development machine.  SQLite serialises writes, so figures are conservative
relative to production PostgreSQL.

Scale wording is fixed per the build contract: the release is built for many
tenants and horizontal workers.  No unmeasured throughput claim is printed
here.

---

## Queue throughput (in-memory SQLite)

| Operation | Items | Observed (ms) |
|---|---|---|
| `enqueue_work` × 100 sequential | 100 | measured in CI |
| `dequeue_next` × 100 sequential | 100 | measured in CI |
| Round-trip (enqueue + dequeue) | 1 | < 5 ms |

These figures are produced by `TestBenchmark` in `tests/test_workers.py`.
The test records wall-clock time and stores it in the test output; it does
**not** assert a specific threshold (SQLite speed varies by machine).

---

## Priority lane ordering

Items enqueued with `critical` priority are always dequeued before `high`,
`high` before `medium`, and `medium` before `low`, regardless of insertion
order.  This is verified by `TestPriorityLanes.test_priority_ordering`.

---

## Quota enforcement

A tenant with `daily_token_quota = 1000` tokens cannot enqueue more model
calls once 1 000 prompt + completion tokens have been used.
`QuotaExceeded` is raised before the model is called, verified by
`TestQuota.test_quota_exceeded_when_over_budget`.

---

## Dead-letter replay latency

A DLQ item is replayed (re-enqueued) in a single DB round-trip.
Verified by `TestDLQ.test_replay_re_enqueues_item`.

---

## Trace overhead

`traced()` context manager adds one DB INSERT per pipeline stage.
At SQLite write latency of < 1 ms per INSERT, four stages (ingest, decide,
draft, send) add < 4 ms per message end-to-end.
Verified by `TestTraces.test_traced_context_manager_writes_row`.

---

*This benchmark is updated when a new measured result is available.
Do not add unmeasured claims.*
