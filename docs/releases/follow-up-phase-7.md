# Follow-up Phase 7 — Company capabilities (fifteen buyer options)

**Product:** Buro Assistant  
**Contract:** Follow-up contract (development restart 2026-10-04)  
**Phase:** 7 — Company capabilities (fifteen buyer options, Section B)  
**Date:** 2026-10-08  
**Branch:** main  

---

## What this phase does

Implements all fifteen buyer options from Section B of DIRECTIVE.txt as
working code with passing tests.  No option is README-only text.

```
app/domain/buyer_features.py   – all fifteen features in one module
migrations/versions/0010_buyer_features.py  – eleven new DB tables
tests/conftest.py              – buyer_features models registered on Base
tests/test_buyer_features.py   – 50 new tests (B1–B15)
README.md                      – roadmap updated (Phase 7 status → closed)
```

---

## Section B implementation summary

| # | Option | Implementation | Table(s) |
|---|---|---|---|
| B1 | Shared team inbox with assignment | `assign_message()`, `get_assignment()` | `message_assignments` |
| B2 | Internal note (never sent) | `add_internal_note()`, `notes_for_message()` | `internal_notes` |
| B3 | Collision lock for drafts | `acquire_draft_lock()`, `release_draft_lock()`, `get_draft_lock()` | `draft_locks` |
| B4 | Snooze until a business hour | `snooze_message()`, `due_snoozed_messages()` | `snoozed_messages` |
| B5 | VIP sender list per tenant | `add_vip_sender()`, `is_vip_sender()` | `vip_senders` |
| B6 | After-hours holding queue | `hold_after_hours()` (wraps B4 with reason="after_hours") | `snoozed_messages` |
| B7 | Saved reply snippets (template-checked) | `save_reply_snippet()`, `list_snippets()` — forbidden-phrase check enforced | `reply_snippets` |
| B8 | Per-department SLA | `set_department_sla()`, `get_department_sla_config()` — returns dict compatible with `policy.sla` | `department_sla` |
| B9 | CSV export of audit log | `csv_export_audit()` — UTF-8 CSV, tenant-scoped | no new table |
| B10 | Bounce and failure reason on desk | `record_delivery_failure()`, `failures_for_tenant()` | `delivery_failures` |
| B11 | Vacation responder (no invented facts) | `set_vacation_responder()`, `vacation_responder_active()` — template-only, no model call | `vacation_responders` |
| B12 | Legal hold blocks retention delete | `set_legal_hold()` helper (column already existed from Phase 4) | `messages.legal_hold` |
| B13 | Role split: admin, operator, auditor | `set_user_role()`, `get_user_role()`, `VALID_ROLES` | `user_roles` |
| B14 | German and English receipt templates | `register_language_templates()` — both templates template-checked, no model | no new table |
| B15 | Daily operator digest, shadow by default | `set_digest_subscription()`, `build_digest()` — default send_mode="shadow" | `digest_subscriptions` |

---

## Design constraints enforced

- No feature calls a model.  All new code is persistence, policy, or template logic.
- Tenant isolation: every new table includes a `tenant_id` FK and all helpers filter by it.
- B11 vacation responder: `template_id` must resolve in the template registry; no free-form generation.
- B14 templates: `required_variables` identical to the built-in receipt template; no extra variables that could carry invented content.
- B15 digest: `send_mode="shadow"` is the default; email mode is opt-in.
- B12 legal hold: `set_legal_hold()` is the only API path; direct DB access is still required to clear it, as documented in the threat model.

---

## Test command

```bash
python3 -m pytest tests/ -q
```

**Result:** 504 passed, 0 failed, 0 errors.

Previous baseline: 454 tests (Phase 6 follow-up).  
New tests this phase: 50 (in `tests/test_buyer_features.py`).

---

## Requirements satisfied

| Requirement | Satisfied | Notes |
|---|---|---|
| All 15 Section B options implemented as working code | yes | `app/domain/buyer_features.py` |
| Each option has a passing test | yes | `tests/test_buyer_features.py` — 50 tests, all green |
| No option is README-only | yes | All options have an ORM model or helper function |
| Tenant isolation on all new tables | yes | Every query filters by `tenant_id` |
| No model calls in any buyer feature | yes | Template registry and DB helpers only |
| App stays runnable | yes | 504 tests pass; no existing test broken |

---

## Files changed

| File | Change |
|---|---|
| `app/domain/buyer_features.py` | **New.** All 15 buyer-option models and helpers |
| `migrations/versions/0010_buyer_features.py` | **New.** 11 new tables created idempotently |
| `tests/conftest.py` | `import app.domain.buyer_features` added to model registration block |
| `tests/test_buyer_features.py` | **New.** 50 tests covering all 15 buyer options |
| `README.md` | Roadmap: Phase 7 status updated to "closed" |
| `docs/releases/follow-up-phase-7.md` | **New.** This file |

---

## Known limitations (not failures)

- B3 collision lock: in-process TTL only — no background job evicts stale locks.
  `get_draft_lock()` evicts on read.  A background sweep is planned for Phase 8.
- B9 CSV export: uses the in-memory `io.StringIO` buffer; for very large audit
  logs the operator should use the `limit` parameter or export directly from the DB.
- B11 vacation responder: `active` flag is not automatically cleared when `end_date`
  passes; the operator must deactivate it.
- B13 roles: role enforcement at the HTTP layer (checking the role before serving a
  request) is not yet wired into the desk router; the data model and helpers are in
  place.  Enforcement is planned for Phase 8.

---

## Test failures

None. All 504 tests pass.
