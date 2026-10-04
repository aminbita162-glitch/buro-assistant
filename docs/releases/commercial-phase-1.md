# Commercial Phase 1 — Plan and trial state

**Date:** 2026-10-12
**Branch:** main
**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab

---

## What was built

A tenant subscription record was added to the system.

### New file: `app/domain/subscription.py`

Defines the `TenantSubscription` ORM model and the four subscription states:

| Status | Meaning | `can_route` |
|---|---|---|
| `trial` | Within 10-day trial window | `True` |
| `active` | Paid plan, open routes | `True` |
| `expired` | Trial ended or plan lapsed | `False` |
| `cancelled` | Operator-cancelled | `False` |

Fields on the record:

| Field | Type | Notes |
|---|---|---|
| `tenant_id` | Integer FK | One subscription per tenant (unique) |
| `plan_code` | String | `trial` / `desk` / `mail` / `agents` |
| `status` | String | One of the four states above |
| `trial_end` | DateTime (tz-aware) | Set to `now + 10 days` on creation; `NULL` for paid plans |
| `token_cap` | Integer | Per-plan default; overridable |
| `tokens_used` | Integer | Accumulated; starts at 0 |

Public helpers:

- `create_trial(db, tenant_id)` — creates a trial subscription.
- `get_subscription(db, tenant_id)` — returns the record or `None`.
- `refresh_status(db, sub)` — advances a past-due trial to `expired`.
- `can_route(db, tenant_id)` — returns `True` only for `trial` and `active`.
- `record_tokens(db, tenant_id, count)` — adds to `tokens_used`.

### New migration: `migrations/versions/0011_subscription.py`

Creates the `tenant_subscriptions` table.
Revision chain: `0011` → `0010`.
Idempotent: uses `inspect` to skip creation if the table already exists.

### Updated: `tests/conftest.py`

`app.domain.subscription` is imported at test-setup time so the model is
registered on `Base` before any test creates tables.

---

## Test results

| Run | Command | Result |
|---|---|---|
| New tests only | `python3 -m pytest tests/test_subscription.py -v` | 20 passed, 0 failed |
| Full suite | `python3 -m pytest tests/ -q` | 620 passed, 0 failed |

Test file: `tests/test_subscription.py`

Covers:
- Schema: table exists, all required columns present.
- Trial state (5 tests): create_trial, trial_end window, token_cap default, tokens_used starts at 0, can_route True.
- Expired state (3 tests): refresh_status advances trial, can_route False, no subscription returns False.
- Active state (5 tests): Desk / Mail / Agents can_route True, refresh_status leaves active unchanged, token_cap per plan.
- Cancelled state (2 tests): can_route False, refresh_status leaves cancelled unchanged.
- Token accounting (3 tests): accumulates, multiple calls, get_subscription returns None for unknown tenant.

---

## External test script

`docs/EXTERNAL_TEST.sh` has not been run by a second account.
The row in `docs/TEST_HOUSE.md` is not updated until that run happens.

---

## App runnable

The application starts normally. The migration is applied at startup via
`alembic upgrade head`. If the database is not reachable at startup, the
existing warning-and-continue path applies (no crash).

---

## Scope boundary

Phase 1 ends here. The subscription record exists and the four states are
tested. Plan enforcement (Desk cannot call Amin, etc.) is Phase 3.
The payment gateway is Phase 2. The operator surface is Phase 4.
