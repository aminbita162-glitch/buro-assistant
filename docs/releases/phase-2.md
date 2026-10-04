# Phase 2 — Bounded Thread Context

**Date:** 2026-10-18
**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab
**Branch:** main

---

## Summary

Phase 2 adds a bounded, redacted clip of the last three messages in the same
thread to Amin's model prompt. The clip is capped at 200 characters total.
Attachment bytes never enter the clip. A rule hit still costs zero tokens.

---

## What was built

### New module — `app/agents/thread_context.py`

A self-contained thread-context module that:

- Defines `THREAD_CLIP_CHARS = 200` — the hard cap on the combined clip length.
- Defines `THREAD_LOOKBACK = 3` — maximum number of prior messages to include.
- The public function `fetch_thread_context(db, tenant_id, subject_normalized, exclude_provider_message_id)`:
  - Queries the `messages` table for the last `THREAD_LOOKBACK` messages in the
    same thread (same `tenant_id` + `subject_normalized`), excluding the current
    message by `provider_message_id`.
  - Only messages with `state = "new"` are included; duplicate and quarantine
    rows are excluded.
  - Extracts `body_text` from each row's `raw_json`. Attachment bytes are never
    included.
  - Redacts PII (email, phone, card, IBAN, SSN) from every snippet using the
    existing `app/agents/redact.redact()` function.
  - Joins snippets with ` | ` as a separator. The combined total — including
    separators — is capped at exactly `THREAD_CLIP_CHARS` characters.
  - Returns an empty string when there are no prior messages.
  - Tenant-isolated: messages from other tenants are never returned.

### Triage agent — `app/agents/amin.py`

- `PROMPT_VERSION` incremented from `"amin-v2"` to `"amin-v3"`.
- `triage()` accepts a new optional parameter `thread_context: str = ""`.
- `_build_prompt()` appends a `thread:` line when `thread_context` is non-empty.
  The line appears between `body:` and the `Return JSON:` directive.
- A rule hit skips the model call regardless of whether thread context is
  present — token cost remains zero on every rule-hit path.
- Attachment bytes are never passed to the model (Phase 5 rule retained).

### Pipeline — `app/pipeline.py`

- Imports `fetch_thread_context` from `app/agents/thread_context`.
- Before calling `triage()`, calls `fetch_thread_context()` using the ingested
  message's `tenant_id`, `subject_normalized`, and `provider_message_id`.
- Passes the resulting clip as `thread_context` to `triage()`.

---

## Behaviour contract

| Condition | Thread clip | Model called | Tokens charged |
|---|---|---|---|
| No prior messages in thread | `""` (empty) | if no rule hit | normal |
| Prior messages exist | redacted clip ≤ 200 chars | if no rule hit | normal |
| Rule fires | any clip | **no** | zero |
| Prior messages with `state=duplicate` or `state=quarantine` | excluded | — | — |
| Prior messages from a different tenant | excluded | — | — |
| Attachment bytes in prior messages | excluded | — | — |

---

## Tests added — `tests/test_thread_context.py`

22 new tests across five classes:

| Class | Tests | Covers |
|---|---|---|
| `TestFetchThreadContext` | 13 | Empty result, current message excluded, 1/3 prior messages, only-last-3 limit, 200-char cap, separator-aware cap, duplicate/quarantine exclusion, tenant isolation, PII redaction, missing body |
| `TestBuildPromptThreadContext` | 3 | `thread:` line present when non-empty, absent when empty, appears before `Return JSON:` |
| `TestTriageThreadContext` | 4 | Prompt contains thread context, no thread line when empty, rule hit costs zero tokens, prompt version updated |
| `TestThreadClipConstants` | 2 | `THREAD_CLIP_CHARS == 200`, `THREAD_LOOKBACK == 3` |

### Existing tests updated

- `tests/test_agents.py` — `TestAmin.test_schema_version_and_prompt_version_set`:
  assertion updated from `"amin-v2"` to `"amin-v3"`.
- `tests/test_token_policy.py` — `TestPromptStructure.test_prompt_version_is_amin_v2`:
  assertion updated to `"amin-v3"`.

---

## Test result

`python3 -m pytest tests/ -q` → **747 passed, 0 failed**

(725 prior + 22 new)

---

*For questions contact: Amin Azimi, AI Architect, Azimi Innovation Lab.*
