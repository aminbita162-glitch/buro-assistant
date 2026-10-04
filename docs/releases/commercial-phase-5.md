# Commercial Phase 5 — Record and Freeze

**Date:** 2026-10-16
**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab
**Branch:** main

---

## What was done

Phase 5 of the commercial contract records the completed commercial build and
freezes the state. No new code was written. No new tests were added.

### Summary of phases 1–4

| Phase | Work | Tests after |
|---|---|---|
| Commercial Phase 1 | Tenant subscription record: plan code, status, trial end, token cap, token used. Four subscription states. | 620 passed |
| Commercial Phase 2 | Payment port with fake and live adapters. Gateway secret read from environment only. No secret in repo. | 635 passed |
| Commercial Phase 3 | Plan capability enforcement: desk, mail, agents, trial. Token cap gate. Send block on trial. | 670 passed |
| Commercial Phase 4 | Operator desk `/desk/plan` endpoint. Choose-plan state. Pipeline capability gating. README commercial status. | 691 passed |

### Test run — Phase 5

`python3 -m pytest tests/ -q` — **691 passed, 0 failed**

No new tests were added this phase. The baseline is unchanged from Phase 4.

---

## Files changed

| File | Change |
|---|---|
| `docs/releases/commercial-phase-5.md` | This file |
| `docs/TEST_HOUSE.md` | Phase 5 run recorded |

---

## External test status

`docs/EXTERNAL_TEST.sh` has **not** been run by a second account. The external
test row in `docs/TEST_HOUSE.md` continues to read "not run". No claim is made
that a second account has installed or verified this product.

---

## Freeze

The commercial contract (Phases 1–5) is now frozen on main. No further phases
are started. The product remains:

- No permanent free plan.
- 10-day trial for every new tenant, then a paid plan is required.
- Trial: token cap, no mail send.
- Plan Desk (65 EUR/month), Plan Mail (149 EUR/month), Plan Agents (270 EUR/month).
- No card data stored. Gateway secret from environment only; absent → fake gateway.
- Sale wiring is in progress. Hosting is not offered.
- Apple Developer membership is not a payment rail and is not mentioned.

---

## Honesty note

Second-account external test script (`docs/EXTERNAL_TEST.sh`) has not been run.
Designed-for is not verified. The external test row in `docs/TEST_HOUSE.md`
remains not-run. No throughput figures are stated unless they come from a
measured run.
