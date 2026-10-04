# Commercial Phase 3 — Enforce the three plans

**Date:** 2026-10-14
**Branch:** main
**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab

---

## What was built

Plan-level capability enforcement was added to the system.

### New file: `app/domain/plan_rules.py`

Pure-Python module that accepts `plan_code`, `status`, `tokens_used`, and
`token_cap` as plain values and returns `None` (permitted) or a
`PlanRefusal` (refused) for the requested capability.  No SQLAlchemy session
is required; the module can be called from any layer.

#### Capability names

| Constant | Value | Meaning |
|---|---|---|
| `CAP_AGENT_AMIN` | `"agent:amin"` | Route to Amin |
| `CAP_AGENT_AMILOS` | `"agent:amilos"` | Route to Amilos |
| `CAP_AGENT_LEILA` | `"agent:leila"` | Route to Leila |
| `CAP_MODEL` | `"model"` | Call an LLM |
| `CAP_SEND` | `"send"` | Send outbound mail |

#### Enforcement rules

| Plan | Capability | Result |
|---|---|---|
| `desk` | `agent:amin`, `agent:amilos`, `agent:leila` | `PlanRefusal("desk_plan_no_agents")` |
| `desk` | `model` | `PlanRefusal("desk_plan_no_model")` |
| `mail` | `model` | `PlanRefusal("mail_plan_no_model")` |
| `agents` | any agent or model, `tokens_used >= token_cap` | `PlanRefusal("agents_plan_token_cap_reached")` |
| `trial` | any agent or model, `tokens_used >= token_cap` | `PlanRefusal("trial_token_cap_reached")` |
| any plan | `send`, when `status == trial` | `PlanRefusal("trial_send_blocked")` |
| `expired` or `cancelled` | any capability | `PlanRefusal("subscription_not_active")` |

All other combinations return `None` (capability permitted).

**A rule hit costs zero tokens.**  The calling layer must not invoke
`record_tokens()` when `check_plan_capability()` returns a `PlanRefusal`.

#### `PlanRefusal` semantics

- `bool(PlanRefusal(...))` is `False` — callers can write `if check_plan_capability(...):`.
- `repr()` includes the reason string.
- `None` (allow) is truthy; `PlanRefusal` (refuse) is falsy.

#### Public interface

```
check_plan_capability(
    capability: str,
    plan_code: str,
    status: str,
    tokens_used: int = 0,
    token_cap: int = 0,
) -> Optional[PlanRefusal]
```

---

## Test results

| Run | Command | Result |
|---|---|---|
| New tests only | `python3 -m pytest tests/test_plan_rules.py -v` | 35 passed, 0 failed |
| Full suite | `python3 -m pytest tests/ -q` | 670 passed, 0 failed |

Test file: `tests/test_plan_rules.py`

Covers:

- **Desk plan** (7 tests): Amin refused, Amilos refused, Leila refused, model
  refused, send allowed when active, `PlanRefusal` is falsy, rule hit costs
  zero tokens (refused at any token level).
- **Mail plan** (6 tests): model refused (absolute, not token-gated), Amin
  allowed, Amilos allowed, Leila allowed, send allowed when active, model
  refused regardless of token count.
- **Agents plan** (11 tests): Amin/Amilos/Leila/model allowed under cap, all
  three agents and model refused at cap, Amin refused over cap, send allowed,
  one token below cap is still allowed.
- **Trial** (6 tests): Amin and model allowed under cap, Amin and model refused
  at cap, send refused (trial blocks send regardless of token count).
- **Inactive subscriptions** (2 tests): expired and cancelled refuse every
  capability with reason `"subscription_not_active"`.
- **`PlanRefusal` semantics** (3 tests): falsy, `repr()` contains reason,
  `None` (allow) is truthy.

All tests are pure-Python; no database session is required.

---

## External test script

`docs/EXTERNAL_TEST.sh` has not been run by a second account.
The row in `docs/TEST_HOUSE.md` is not updated until that run happens.

---

## App runnable

The application starts normally.  `app/domain/plan_rules.py` is a pure-Python
module with no migration and no new database table.  Callers that do not yet
invoke `check_plan_capability()` are unaffected.

---

## Scope boundary

Phase 3 ends here.  The three plan rules are implemented and all 35 tests pass.
The operator surface (plan display, trial days left, choose-plan state) is Phase 4.
