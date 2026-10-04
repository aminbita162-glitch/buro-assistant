# Phase 5 — Audit Hash Chain

**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab
**Date:** 2026-10-21
**Branch:** main

---

## Summary

Phase 5 extends the append-only audit log with a SHA-256 hash chain.
Every new row stores the hash of the previous row. A changed row breaks
the chain. No old rows were rewritten.

---

## What was built

### `app/policy/audit.py`

- Added `prev_hash` column (`String(64)`, nullable) to `AuditLogEntry`.
  Rows written before Phase 5 carry `NULL`; they are excluded from chain
  verification.
- `log_event()` now computes `prev_hash` before each insert:
  - Queries the most recent chained row for the same tenant (ordered by
    `id desc`, `prev_hash IS NOT NULL`).
  - If no such row exists, writes the sentinel `ZERO_HASH`
    (`"000…000"`, 64 zeros).
  - Otherwise writes `SHA-256(_canonical(prev_entry))`.
- `_canonical(entry)` produces a stable pipe-separated string from
  `id | tenant_id | event | actor | detail | created_at.isoformat()`.
- `verify_chain(db, tenant_id)` reads all chained rows in insertion order
  and returns the ids of any row whose stored `prev_hash` does not match
  the expected value. An empty list means the chain is intact.
- Added `ZERO_HASH` constant and internal helpers `_canonical`, `_sha256`.
- No update or delete function was added.

### `migrations/versions/0013_audit_hash_chain.py`

- Adds `prev_hash String(64) nullable` to `audit_log` if the table
  exists and the column is absent.
- Safe for existing databases: existing rows remain with `NULL`.
- Includes a `downgrade()` that drops the column via batch alter
  (required for SQLite compatibility).

---

## Tests — `tests/test_audit_hash_chain.py`

18 new tests, all passing.

| Class | Tests | What is verified |
|---|---|---|
| `TestHashChainWrite` | 5 | First row carries `ZERO_HASH`; second row carries `SHA-256(row1)`; third row carries `SHA-256(row2)`; `prev_hash` is 64 valid hex chars; `ZERO_HASH` is 64 zeros |
| `TestVerifyChainIntact` | 3 | Empty log, single row, and five-row chain all return `[]` |
| `TestVerifyChainBroken` | 6 | Tampered `event`, `detail`, or `actor` each break the next row; tampering row 2 of four rows reports row 3 (not row 1 or row 4); directly replacing `prev_hash` is detected; untampered rows are not reported |
| `TestPreChainRowsExcluded` | 2 | A row with `NULL` `prev_hash` is skipped; new rows after a legacy row start their chain from `ZERO_HASH` |
| `TestTenantChainIsolation` | 2 | Two tenants have independent chains; tampering one tenant does not affect the other |

---

## Test run

| # | Date | Actor | Action | Result |
|---|---|---|---|---|
| 21 | 2026-10-21 | Amin Azimi | `python3 -m pytest tests/ -q` | 797 passed, 0 failed |

---

## Honesty note

The `prev_hash` column is nullable. Rows written before Phase 5 have
`NULL` and are excluded from verification. This is by design: the
directive says "do not rewrite old rows". The chain starts from the
first row written after Phase 5 is deployed.

---

## Files changed

| File | Change |
|---|---|
| `app/policy/audit.py` | Added `prev_hash` column, hash chain logic, `verify_chain`, helpers |
| `migrations/versions/0013_audit_hash_chain.py` | Migration: add `prev_hash` to `audit_log` |
| `tests/test_audit_hash_chain.py` | 18 new tests |
| `docs/TEST_HOUSE.md` | Run 21 recorded |
| `docs/releases/final-phase-5.md` | This file |
