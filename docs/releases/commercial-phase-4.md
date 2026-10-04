# Commercial Phase 4 — Operator Surface

**Date:** 2026-10-15
**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab
**Branch:** main

---

## What was done

Phase 4 of the commercial contract adds the operator-facing plan surface and wires
plan capability enforcement into the agent pipeline.

### 1. `/desk/plan` endpoint

`GET /desk/plan` (Bearer auth required) returns the tenant's current commercial state:

| Field | Type | Description |
|---|---|---|
| `plan_code` | string | `"trial"` \| `"desk"` \| `"mail"` \| `"agents"` |
| `status` | string | `"trial"` \| `"active"` \| `"expired"` \| `"cancelled"` |
| `trial_days_left` | integer | Days remaining in trial window; 0 when not on trial |
| `token_cap` | integer | Per-billing-period token cap |
| `tokens_used` | integer | Tokens consumed this billing period |
| `choose_plan` | boolean | `true` when status is expired or cancelled — desk shows choose-plan state |
| `plans_available` | list | Desk / Mail / Agents plan catalogue with EUR prices |

The desk has **no card field**. No card data is stored or returned anywhere.

The endpoint does not lock the whole app when the trial is expired. It sets
`choose_plan: true` and the operator desk can render a choose-plan state without
blocking other desk functionality.

### 2. Plan catalogue

Three plans are listed (no card field, no payment collection in this phase):

| Code | Name | Price | Band |
|---|---|---|---|
| `desk` | Desk | 65 EUR/month | 50–80 |
| `mail` | Mail | 149 EUR/month | 80–200 |
| `agents` | Agents | 270 EUR/month | 200–320 |

Sale wiring is in progress. Hosting is not offered.

### 3. `check_plan_capability` wired into the pipeline

`app/pipeline.py` now calls `check_plan_capability` at two points before any
agent work:

1. **Before Amin (triage)** — checks `CAP_AGENT_AMIN`. When a `triage_model`
   argument is passed, also checks `CAP_MODEL`.
2. **Before Amilos (draft)** — checks `CAP_AGENT_AMILOS` when the triage
   action is `draft_reply`.

A refusal produces `outcome="plan_refused"` with the refusal reason in `note`.
Zero tokens are recorded; the cost event is not emitted on a plan-refused path
before triage. On the post-triage Amilos refusal, triage cost is still emitted
if non-zero.

No subscription row is treated the same as an expired subscription: the pipeline
returns `plan_refused`.

### 4. README

Added a **Commercial status** section stating:
- Sale wiring is in progress.
- Hosting is not offered.

---

## Files changed

| File | Change |
|---|---|
| `app/web/desk.py` | Added `GET /desk/plan` endpoint and `_plan_catalogue()` helper |
| `app/pipeline.py` | Added `check_plan_capability` gating before Amin and Amilos calls |
| `tests/test_operator_surface.py` | 25 new tests (auth guard, shape, choose-plan, pipeline gating) |
| `README.md` | Added Commercial status section; updated Roadmap |
| `docs/releases/commercial-phase-4.md` | This file |
| `docs/TEST_HOUSE.md` | Phase 4 run recorded |

---

## Tests

21 new tests in `tests/test_operator_surface.py`.

Previous baseline: 670 tests.
After Phase 4: 691 tests.

Run: `python3 -m pytest tests/ -q`

---

## Not done in this phase

- Payment collection is not wired. The payment port from Phase 2 exists but
  `/desk/plan` does not call it and does not return a card field.
- No UI is shipped. The endpoint returns JSON for the operator desk to render.
- Apple Developer membership is not mentioned anywhere in this phase.

---

## Honesty note

Second-account external test script (`docs/EXTERNAL_TEST.sh`) has not been run.
The external test row in `docs/TEST_HOUSE.md` remains not-run.
