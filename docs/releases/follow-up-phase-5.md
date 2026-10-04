# Follow-up Phase 5 — lowest-token policy, triage cache, and dashboard cost counters

**Product:** Buro Assistant  
**Contract:** Follow-up contract (development restart 2026-10-04)  
**Phase:** 5 – lowest-token policy, triage cache, and dashboard cost counters  
**Date:** 2026-10-07  
**Branch:** main  

---

## What this phase does

Extends the Phase 5 three-agent pipeline with four targeted improvements required
by the DIRECTIVE.txt stop conditions that were not included in the original Phase 5
release:

```
Token policy     app/agents/amin.py
                   – BODY_CLIP_CHARS = 200: only the first 200 chars of the
                     redacted body are sent to the model; full body is truncated
                   – Attachment bytes are never passed to the model (enforced by
                     construction: only msg.subject and msg.body_text are used)
                   – Prompt version bumped to "amin-v2" to mark the format change

Triage cache     app/agents/amin.py
                   – In-process dict cache (_TRIAGE_CACHE) keyed on
                     (redacted_subject, body_clip, pack_hash, rule_hit_name)
                   – Rule-hit results are deterministic; cache avoids redundant work
                   – Bounded at 512 entries; oldest entry evicted on overflow

Rule pack hash   app/agents/rules.py
                   – rule_pack_hash(): SHA-256 of domain_rules + subject_rules +
                     department_rules serialised as canonical JSON; first 16 hex chars
                   – Hash embedded in every model prompt so rule-pack changes are
                     traceable in prompt history without leaking rule content

Dashboard        app/web/desk.py  +  app/pipeline.py
                   – run_pipeline() calls _emit_cost_event() on every path
                     (draft_reply path and hold/approval path)
                   – _emit_cost_event() writes a usage_event row with unit="tokens"
                   – GET /desk/cost response extended with four new fields:
                       today_tokens_used, today_cost_usd,
                       daily_token_quota, today_tokens_remaining
                   – Quota row for today is read via get_or_create_quota() so the
                     dashboard shows live daily usage in one API call
```

---

## Test command

```bash
python3 -m pytest tests/ -q
```

**Result:** 454 passed, 0 failed, 0 errors.

Previous baseline: 440 tests (Phase 4 follow-up).  
New tests this phase: 14 (in `tests/test_token_policy.py`).

---

## Requirements satisfied

| Requirement | Satisfied | Notes |
|---|---|---|
| Rule before model — zero model tokens on rule hit | yes | `triage()` returns from cache or builds decision before any `model.call()` |
| Attachment bytes never sent to model | yes | `_build_prompt` receives only `redacted_subject` and `body_clip`; attachments not referenced |
| Prompt body clipped to ≤ 200 chars | yes | `body_clip = redacted_body[:BODY_CLIP_CHARS]` before prompt assembly |
| Rule pack hash in every model prompt | yes | `rule_pack_hash()` result prefixed as `rules_hash:` in prompt |
| Prompt version reflects format change | yes | `PROMPT_VERSION = "amin-v2"` |
| Triage cache: identical rule-hit inputs skip model | yes | `_TRIAGE_CACHE` dict; bounded at 512 entries; model-call results not cached |
| Pipeline emits usage_event on model call | yes | `_emit_cost_event()` called on draft and non-draft paths |
| Rule-hit path emits zero-token usage event | yes | `null_cost()` passed to `_emit_cost_event()` when no model called |
| `/desk/cost` returns today's token counters | yes | `today_tokens_used`, `today_cost_usd`, `daily_token_quota`, `today_tokens_remaining` |

---

## Files changed

| File | Change |
|---|---|
| `app/agents/amin.py` | `BODY_CLIP_CHARS = 200`; `_TRIAGE_CACHE` dict + `_CACHE_MAX = 512`; `PROMPT_VERSION` bumped to `"amin-v2"`; `rule_pack_hash` imported and embedded in prompt; cache lookup and store around rule-hit path; `_build_prompt` signature extended with `pack_hash`; attachment-safety comment added. |
| `app/agents/rules.py` | `rule_pack_hash()` function added: `hashlib` + `json` imported; SHA-256 of canonical rule-pack fields, returns 16-char hex prefix. |
| `app/pipeline.py` | `emit as emit_usage` imported from `app.domain.usage`; `_emit_cost_event()` helper added (best-effort, never raises); called on both the `draft_reply` path and the hold/approval path. |
| `app/web/desk.py` | `GET /desk/cost` docstring updated; `get_or_create_quota` + `DEFAULT_DAILY_TOKEN_QUOTA` imported; `today_tokens_used`, `today_cost_usd`, `daily_token_quota`, `today_tokens_remaining` added to response dict. |
| `tests/test_agents.py` | `prompt_version` assertion updated from `"amin-v1"` to `"amin-v2"`. |
| `tests/test_token_policy.py` | **New.** 14 tests covering: rule-hit zero tokens, attachment exclusion, prompt structure (hash, body clip, redacted subject, version), triage cache, and dashboard usage event emission. |
| `docs/releases/follow-up-phase-5.md` | **New.** This file. |

---

## Known limitations (not failures)

- `_TRIAGE_CACHE` is in-process only; it does not survive a worker restart and is
  not shared across multiple worker processes.  This is intentional: the cache is a
  performance optimisation, not a correctness dependency.
- Model-call results are not cached.  Only deterministic rule-hit results are stored.
  Non-rule-hit messages always call the model to allow confidence to vary.
- `_emit_cost_event()` is best-effort: any database error inside the helper is
  silently swallowed so that a usage-logging failure cannot break the pipeline.

---

## Test failures

None. All 454 tests pass.
