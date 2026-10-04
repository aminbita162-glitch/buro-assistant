# Phase 6 — Desk Signals

**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab
**Date:** 2026-10-22
**Branch:** main

---

## Summary

Phase 6 adds four desk signals to the pipeline: a daily digest stored as
shadow only, a semantic-duplicate flag on inbound messages, attachment text
extraction from already-allowed types, and a dissatisfied-tone flag that
routes to Leila and blocks auto-send.

---

## What was built

### `app/policy/digest.py`

New module. Public API: `store_digest_shadow(db, tenant_id) -> Draft`.

- Builds a daily digest payload: counts of messages received, drafts created,
  pending approval entries, semantic-duplicate flags, and dissatisfied-tone
  flags for the tenant.
- Persists the payload as a `Draft` row with `state="shadow"` and
  `template_id="daily_digest"`. Draft body is the JSON-serialised payload.
- Never calls a mail sender. Never calls a model. No attachment bytes are
  passed anywhere. The state is always `"shadow"` — there is no code path
  that changes it.

### `app/ingest/models.py`

Two boolean columns added to the `Message` model:

| Column | Type | Default | Purpose |
|---|---|---|---|
| `semantic_duplicate` | `Boolean` | `False` | Near-duplicate body detected; flag only — no second row created |
| `dissatisfied_tone` | `Boolean` | `False` | Dissatisfaction signals detected; route to Leila, do not send |

### `app/ingest/attachment_text.py`

New module. Public API:

- `allowed_for_extraction(content_type) -> bool` — returns `True` when the
  content-type is in the existing `ATTACHMENT_ALLOWLIST`.
- `extract_text(content_type, data: bytes) -> str` — decodes only `text/plain`
  and `text/csv` bytes to a string. Returns `""` for `application/pdf`,
  images, and binary office formats. Returns `""` for any disallowed type.
  Never raises. The returned string must not be forwarded to a model.

Rules enforced:
- Only already-allowed attachment types are accepted (no new allowlist).
- PDF and binary office formats are not decoded (binary parsing not performed).
- Image pixels are not decoded.
- The returned value is always `str`, never `bytes`.

### `app/agents/signals.py`

New module. Public API:

- `is_semantic_duplicate(body, candidates, threshold=0.85) -> bool` — Jaccard
  word-set similarity. Returns `True` when any candidate reaches the threshold.
  No model call.
- `has_dissatisfied_tone(text) -> bool` — keyword lexicon check. Returns `True`
  when any dissatisfaction term is found (case-insensitive substring). No model
  call.

The dissatisfied-tone lexicon includes 30 terms covering phrases such as
`"completely unacceptable"`, `"formal complaint"`, `"demand a refund"`,
`"very disappointed"`, `"extremely unhappy"`.

### `migrations/versions/0014_desk_signals.py`

- Adds `semantic_duplicate Boolean NOT NULL DEFAULT FALSE` to `messages`.
- Adds `dissatisfied_tone Boolean NOT NULL DEFAULT FALSE` to `messages`.
- Safe for existing databases: checks for column existence before adding.
- `downgrade()` drops both columns via batch alter (SQLite compatible).

---

## Tests — `tests/test_desk_signals.py`

45 new tests, all passing.

| Class | Tests | What is verified |
|---|---|---|
| `TestDailyDigestShadow` | 9 | Returns a Draft; state is always "shadow"; template_id is "daily_digest"; body is valid JSON with required keys; persisted to drafts table; counts messages correctly; tenant-scoped; multiple digests all shadow |
| `TestSemanticDuplicateFlag` | 9 | Identical bodies flagged; near-identical bodies flagged; different bodies not flagged; empty candidates return False; threshold respected; default False on new message; flag can be set; no second message created when flagged; digest counts the flags |
| `TestAttachmentTextExtract` | 14 | allowed_for_extraction: text/plain, text/csv, pdf pass; zip/exe fail; extract_text: text/plain and csv decoded; PDF returns ""; images return ""; disallowed returns ""; return type always str; bad UTF-8 handled; docx returns ""; case-insensitive content-type |
| `TestDissatisfiedToneFlag` | 13 | Multiple positive phrases detected; neutral and empty text not flagged; case-insensitive; default False; flag can be set; routes to Leila with hold/request_human/reroute action; Leila never returns "send"; digest counts the flags; send blocked when flag is True |

---

## Test run

| # | Date | Actor | Action | Result |
|---|---|---|---|---|
| 22 | 2026-10-22 | Amin Azimi | `python3 -m pytest tests/ -q` | 842 passed, 0 failed |

---

## Honesty note

- `extract_text` does not parse PDF or binary office formats. Those types are
  in the allowlist but yield an empty string. No binary-to-text library is used.
- `is_semantic_duplicate` uses a word-set Jaccard heuristic, not an embedding
  model. Semantically similar messages that use different words may not be caught.
- `has_dissatisfied_tone` uses a fixed keyword lexicon. Messages that express
  dissatisfaction in language not covered by the lexicon will not be flagged.
- Both flags are set by the caller after ingest, not automatically by
  `ingest_message`. The pipeline layer or worker must apply them.
- The digest send_mode is always `"shadow"` in `store_digest_shadow`. The
  existing `DigestSubscription` table (from buyer_features B15) is separate
  and is not modified here.

---

## Files changed

| File | Change |
|---|---|
| `app/policy/digest.py` | New: daily digest shadow store |
| `app/ingest/models.py` | Added `semantic_duplicate` and `dissatisfied_tone` columns |
| `app/ingest/attachment_text.py` | New: attachment text extraction (allowed types only) |
| `app/agents/signals.py` | New: `is_semantic_duplicate`, `has_dissatisfied_tone` |
| `migrations/versions/0014_desk_signals.py` | Migration: add two columns to messages |
| `tests/test_desk_signals.py` | 45 new tests |
| `docs/TEST_HOUSE.md` | Run 22 recorded |
| `docs/releases/final-phase-6.md` | This file |
