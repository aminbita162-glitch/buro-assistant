# Follow-up Phase 1 — Wire the pipeline

**Product:** Buro Assistant  
**Contract:** Follow-up contract (development restart 2026-10-04)  
**Phase:** 1 – Wire the pipeline  
**Date:** 2026-10-05  
**Branch:** main  

---

## What this phase does

Connects the existing building blocks into an end-to-end pipeline:

```
ingest_message → Amin (triage) → Amilos (draft) or Leila (supervise)
                               → shadow store or approval queue
```

---

## Test command

```bash
python3 -m pytest tests/ -q
```

**Result:** 370 passed, 0 failed, 0 errors.

New tests: `tests/test_pipeline.py` — 19 tests.

---

## Requirements satisfied

| Requirement | Satisfied | Notes |
|---|---|---|
| Ingest connected to Amin | yes | `run_pipeline()` calls `ingest_message()` then `triage()` |
| Amin connected to Amilos or Leila | yes | `draft_reply` action → Amilos; all other actions use Leila-routed or supervisor decision |
| Rule hit must not call a model | yes | `_TrackingModel.called` stays False; test asserts `model.calls == []` |
| Tokens and cost recorded only on model call | yes | `null_cost()` returned on rule-hit paths; `_TrackingModel.as_cost_record()` populated only when `.called` is True |
| Send off unless policy allows | yes | `should_send()` checked; outcome `"send"` only when `auto_reply_enabled: True` |
| Shadow mode stores draft, not sent | yes | `store_draft()` called with `policy_config`; Draft state is `"shadow"` |
| Duplicate stops pipeline before agent call | yes | `ingest_result == "duplicate"` returns early |
| Quarantine stops pipeline before agent call | yes | `ingest_result == "quarantine"` returns early |
| Non-draft actions enqueue for human review | yes | `hold`, `escalate`, `request_human` → `enqueue_approval()` |
| Reject produces no draft and no approval entry | yes | `outcome == "no_draft"`, both IDs None |
| Low-confidence routes through Leila | yes | Amin already delegates to Leila; pipeline receives supervisor decision and routes to approval |
| App stays runnable | yes | `app/main.py` and all routes unchanged |

---

## Files changed

| File | Change |
|---|---|
| `app/pipeline.py` | **New.** `run_pipeline()` function connecting all pipeline stages. `_TrackingModel` wrapper for cost accounting. `PipelineResult` dataclass. |
| `tests/test_pipeline.py` | **New.** 19 chain tests covering all requirements above. |
| `docs/releases/follow-up-phase-1.md` | **New.** This file. |

---

## Known limitations (not failures)

- `run_pipeline()` does not write a Trace row (worker trace integration is a Phase 2+ concern once the live worker loop exists).
- Cost token counts on the rule-hit path are zero; the `model_name` field defaults to `"fake"`. This is correct — no model was called.
- The receipt template is used for all `draft_reply` actions regardless of department. A per-department template selector is out of scope for Phase 1.

---

## Test failures

None. All 370 tests pass.
