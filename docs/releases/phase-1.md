# Phase 1 — Sender Authentication

**Date:** 2026-10-17
**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab
**Branch:** main

---

## Summary

Phase 1 adds SPF, DKIM, and DMARC result storage to every inbound message.
Amin can read the results. A fail does not auto-send and does not delete.
If live DNS is not configured, the check is recorded as `not_run`.

---

## What was built

### New module — `app/ingest/sender_auth.py`

A self-contained sender-authentication module that:

- Defines `SenderAuthResult` with three fields: `spf`, `dkim`, `dmarc`.
- Each field holds exactly one of `"pass"`, `"fail"`, or `"not_run"`.
- `SenderAuthResult.any_fail` returns `True` when at least one field is `"fail"`.
- The public function `check_sender_auth(sender, raw_headers)` runs DNS lookups
  via `dnspython` when `SENDER_AUTH_ENABLED=1` is set in the environment.
- When `SENDER_AUTH_ENABLED` is absent or `"0"`, all three checks return `"not_run"`.
- When `dnspython` is not installed, all three checks return `"not_run"`.
- SPF: checks the `v=spf1` TXT record; `-all` (hard fail) → `"fail"`.
- DKIM: checks for a `DKIM-Signature` header and resolves the selector key.
- DMARC: checks for a `v=DMARC1` TXT record at `_dmarc.<domain>`.

### Schema changes — `app/ingest/models.py`

Three new `String` columns added to the `messages` table:

| Column | Values | Default |
|---|---|---|
| `auth_spf` | `pass` / `fail` / `not_run` | `not_run` |
| `auth_dkim` | `pass` / `fail` / `not_run` | `not_run` |
| `auth_dmarc` | `pass` / `fail` / `not_run` | `not_run` |

### NormalizedMessage — `app/ingest/normalize.py`

Three new fields `auth_spf`, `auth_dkim`, `auth_dmarc` added to
`NormalizedMessage` (default `"not_run"`). Provider adapters or callers may
pre-populate these; the ingest layer uses pre-populated values when present
and only runs the DNS check when all three are `"not_run"`.

### Ingest layer — `app/ingest/ingest.py`

`ingest_message` now runs `check_sender_auth` (or uses pre-populated values)
and stores all three results on the `Message` row. A `"fail"` result does
**not** change the message state; the message is always stored.

### Triage agent — `app/agents/amin.py`

The triage decision dict returned by `triage()` now includes:

```json
{
  "auth_spf": "pass | fail | not_run",
  "auth_dkim": "pass | fail | not_run",
  "auth_dmarc": "pass | fail | not_run"
}
```

The in-process triage cache key was extended to include auth fields so that
two messages with the same subject but different auth results are never confused.

### Pipeline — `app/pipeline.py`

Before issuing the `send` outcome, the pipeline checks `triage_decision` for
any `"fail"` auth result. If any check is `"fail"`, auto-send is blocked
regardless of the tenant `auto_reply_enabled` policy. The message is preserved
in the database; it is **not** deleted.

### Migration — `migrations/versions/0012_sender_auth.py`

Revision `0012` (revises `0011`) adds `auth_spf`, `auth_dkim`, `auth_dmarc`
to the `messages` table with `server_default="not_run"`. Existing rows receive
`"not_run"` on upgrade.

---

## Behaviour contract

| Condition | SPF/DKIM/DMARC stored | Auto-send | Message deleted |
|---|---|---|---|
| DNS check passes | `"pass"` | allowed if policy permits | no |
| DNS check fails | `"fail"` | **blocked** | no |
| DNS not configured (`SENDER_AUTH_ENABLED` absent) | `"not_run"` | allowed if policy permits | no |
| `dnspython` not installed | `"not_run"` | allowed if policy permits | no |

---

## Tests added — `tests/test_sender_auth.py`

34 new tests across six classes:

| Class | Tests | Covers |
|---|---|---|
| `TestSenderAuthResult` | 8 | Value validation, `any_fail`, sentinel constant |
| `TestCheckSenderAuthDisabled` | 4 | Not-run when env unset, bad sender, dnspython absent |
| `TestCheckSenderAuthPass` | 4 | SPF pass, DMARC pass, DKIM pass, all pass |
| `TestCheckSenderAuthFail` | 4 | SPF fail, DKIM fail, DMARC not-run on error, `any_fail` |
| `TestIngestStoresAuthColumns` | 5 | DB columns present, pass/fail/not_run stored, fail preserves message |
| `TestPipelineAuthSendBlock` | 6 | SPF/DKIM/DMARC fail blocks send, pass and not_run allow send, message preserved |
| `TestTriageDecisionIncludesAuth` | 3 | Auth fields in triage decision for pass, fail, not_run |

---

## Test result

`python3 -m pytest tests/ -q` → **725 passed, 0 failed**

---

*For questions contact: Amin Azizi, AI Architect, Azimi Innovation Lab.*
