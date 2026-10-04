# Follow-up Phase 2 — Live intake and delivery

**Product:** Buro Assistant  
**Contract:** Follow-up contract (development restart 2026-10-04)  
**Phase:** 2 – Live intake and delivery  
**Date:** 2026-10-05  
**Branch:** main  

---

## What this phase does

Wires a poll-based intake worker loop and a signed webhook delivery layer with
retries and a delivery log:

```
IntakeLoop.start()
  └─ run_once(db, tenant_id, provider)
       ├─ FakeProvider (no IMAP_HOST)  or  IMAPProvider (IMAP_HOST set)
       ├─ run_pipeline(db, msg, ...)   →  outcome
       └─ on exception → send_to_dlq(...)

dispatch_event(db, tenant_id, event_type, data, secrets_map)
  └─ deliver_event(db, sub, event_type, data, secret)
       ├─ build_signed_payload(...)  →  body + X-Buro-Signature header
       ├─ _attempt_post(url, payload, sig)  [up to max_attempts]
       └─ _write_log(...)  →  delivery_log row per attempt
```

---

## Test command

```bash
python3 -m pytest tests/ -q
```

**Result:** 396 passed, 0 failed, 0 errors.

New tests: `tests/test_intake.py` — 9 tests.  
New tests: `tests/test_webhook_delivery.py` — 17 tests.

---

## Requirements satisfied

| Requirement | Satisfied | Notes |
|---|---|---|
| Worker loop polls IMAP adapter when IMAP_HOST is set | yes | `_make_provider()` returns `IMAPProvider` when `IMAP_HOST` present |
| Falls back to fake adapter when no mailbox env | yes | `_make_provider()` returns `FakeProvider` when `IMAP_HOST` absent |
| Failed jobs go to the dead-letter queue | yes | `run_once` catches pipeline exceptions and calls `send_to_dlq` |
| Signed webhook payloads delivered with retries | yes | `deliver_event` retries up to `max_attempts` with exponential back-off |
| Delivery log written per attempt | yes | `DeliveryLog` row written for every HTTP attempt (success and failure) |
| Secrets stay in the environment | yes | No secret is stored, logged, or included in any `DeliveryLog` field |
| Tests with the fake adapter | yes | All intake tests use `FakeProvider`; no real IMAP connection |
| App stays runnable | yes | `app/main.py` and all routes unchanged |

---

## Files changed

| File | Change |
|---|---|
| `app/workers/intake_loop.py` | **New.** `run_once()`, `IntakeLoop`, `_make_provider()`, `TickStats`. |
| `app/workers/webhook_delivery.py` | **New.** `deliver_event()`, `dispatch_event()`, `DeliveryLog` ORM, `_write_log()`. |
| `migrations/versions/0007_intake_delivery.py` | **New.** Adds `delivery_log` table. |
| `tests/test_intake.py` | **New.** 9 intake loop tests using `FakeProvider`. |
| `tests/test_webhook_delivery.py` | **New.** 17 webhook delivery and signing tests. |
| `tests/conftest.py` | Register `app.workers.webhook_delivery` ORM model on `Base`. |
| `docs/releases/follow-up-phase-2.md` | **New.** This file. |

---

## Known limitations (not failures)

- `IntakeLoop` is single-tenant per instance. Running multiple tenants requires
  multiple instances (one per tenant) or a supervisor-level wrapper; that is
  out of scope for this phase.
- `dispatch_event` requires the caller to supply a `secrets_map` mapping
  subscription id → plaintext secret. Since plaintext secrets are shown once at
  creation and not persisted, the operator must store them in the environment
  or a secrets store. This is by design (secrets are never in the DB).
- HTTP delivery in `_attempt_post` uses `urllib.request` (stdlib only, no
  dependency added). A production deployment may substitute `httpx` or
  `requests` by patching `_attempt_post`; the interface is stable.

---

## Test failures

None. All 396 tests pass.
