# Phase 6 Release Notes

**Product:** Buro Assistant
**Release tag:** phase-6
**Date:** 2026-10-03
**Checklist rows closed:** 16, 17, 18, 19, 20, 21, 22, 48

---

## Summary

Phase 6 delivers the policy and receipt layer: a template registry with variable substitution and forbidden-phrase checking, a send-only-on-allow gate that keeps auto-reply off by default, a receipt template confirming receipt and review, per-tenant business-hours calendars, an SLA clock from ingest time, a human approval queue, an append-only audit log, and shadow mode that stores drafts without sending. Three new database tables are created by migration `0004`.

---

## What was built

### `app/policy/templates.py` — template registry (rows 16, 18)

- `TemplateEntry` dataclass: `id`, `subject`, `body`, `required_variables` (set), `forbidden_phrases` (frozenset), `language`, `department`.
- `render(variables)` substitutes `{placeholders}`, checks required keys, then scans the rendered body for forbidden phrases and raises `ValueError` on a hit (row 16).
- `TemplateRegistry`: in-memory dict of `template_id → TemplateEntry`. `register()`, `get()`, `require()`, `list_ids()`.
- `DEFAULT_REGISTRY` pre-loaded with `receipt-v1`.
- **`receipt-v1`** (row 18): body text explicitly states "we have received your message and it will be reviewed by our team." Variables: `{original_subject}`, `{sender_name}`, `{reference}`. Forbidden phrases: `guarantee`, `warranty`, `liability`, `no refund`, `not responsible`.

### `app/policy/send_decision.py` — auto-reply gate (rows 17, 48)

- `should_send(policy_config) → "allow" | "shadow" | "draft"`.
- Default (no config, or `auto_reply_enabled` absent/false) → **`"draft"`** (row 17: off unless the tenant turns it on).
- `shadow_mode: true` → **`"shadow"`** — takes priority over `auto_reply_enabled` (row 48).
- `auto_reply_enabled: true` → **`"allow"``.
- `is_send_allowed(policy_config) → bool`: True only when `"allow"`.

### `app/policy/calendar.py` — business-hours calendar (row 19)

- `is_business_hours(dt, calendar)`: returns True if `dt` falls on a working day and within `start_hour ≤ hour < end_hour` (UTC).
- Default calendar: Mon–Fri, 09:00–17:00 UTC.
- Tenant overrides via dict: `working_days`, `start_hour`, `end_hour`, `timezone` (reserved — UTC only at this layer).
- `next_business_start(dt, calendar)`: returns the next business-day opening time at or after `dt`.

### `app/policy/sla.py` — SLA clock (row 20)

- `sla_deadline(ingest_time, urgency, sla_config)`: `ingest_time + sla_hours[urgency]`.
- Default tiers: `critical=1h`, `high=4h`, `medium=24h`, `low=72h`.
- Tenant overrides via `sla_config["sla_hours"]`.
- `is_sla_breached(ingest_time, urgency, …)` and `sla_remaining_seconds(…)` helpers.
- Naive `ingest_time` is treated as UTC.

### `app/policy/approval.py` — human approval queue (row 21)

- `ApprovalQueueEntry` ORM model (`approval_queue` table). Fields: `tenant_id`, `message_id`, `subject`, `body`, `template_id`, `state` (`pending|approved|rejected`), `reason`, `created_at`, `resolved_at`, `resolved_by`.
- `enqueue(db, tenant_id, subject, body, …)`: creates a pending entry.
- `resolve(db, entry_id, tenant_id, decision, resolved_by)`: sets state; cross-tenant resolution returns `None`.
- `pending_for_tenant(db, tenant_id)`: returns all pending entries newest-first.

### `app/policy/audit.py` — append-only audit log (row 22)

- `AuditLogEntry` ORM model (`audit_log` table). Fields: `tenant_id`, `event`, `message_id`, `actor`, `detail`, `created_at`.
- `log_event(db, …)`: the **sole write path** — INSERT only; no update or delete function exists.
- `recent_for_tenant(db, tenant_id, limit)`: read path.
- Event categories: `message_ingested`, `triage_decided`, `draft_created`, `draft_approved`, `draft_rejected`, `message_sent`, `message_held`, `message_failed`, `shadow_draft_stored`, `approval_requested`.

### `app/policy/shadow.py` — shadow mode (row 48)

- `Draft` ORM model (`drafts` table). Fields: `tenant_id`, `message_id`, `subject`, `body`, `template_id`, `language`, `decision_hash`, `state`, `created_at`.
- `store_draft(db, tenant_id, subject, body, …, policy_config)`: persists a draft with state derived from `should_send(policy_config)` — `"shadow"` or `"draft"`; never `"sent"`.
- `is_shadow_mode(policy_config)`: True when `shadow_mode: true`.
- Actual sending (state → `"sent"`) is exclusively the worker layer's responsibility (Phase 8).

---

## Migration

`migrations/versions/0004_policy.py` — creates `drafts`, `approval_queue`, `audit_log` tables. Applied automatically by `_run_migrations()` on startup. All three table creation blocks are idempotent.

---

## Tests

`tests/test_policy.py` — **62 tests** across 8 test classes:

| Class | Tests | Row |
|---|---|---|
| `TestTemplateRegistry` | 9 | 16 |
| `TestReceiptTemplate` | 4 | 18 |
| `TestSendDecision` | 8 | 17 |
| `TestBusinessHoursCalendar` | 8 | 19 |
| `TestSLAClock` | 10 | 20 |
| `TestApprovalQueue` | 8 | 21 |
| `TestAuditLog` | 6 | 22 |
| `TestShadowMode` | 9 | 48 |

Full suite: **210 tests, 0 failures**.

---

## Capability matrix delta

| Row | Capability | Status |
|---|---|---|
| 16 | Template registry with variables and forbidden phrases | ✓ Phase 6 |
| 17 | Auto-reply off unless tenant turns it on | ✓ Phase 6 |
| 18 | Receipt template stating message was received and will be reviewed | ✓ Phase 6 |
| 19 | Business-hours calendar per tenant | ✓ Phase 6 |
| 20 | SLA clock from ingest time | ✓ Phase 6 |
| 21 | Human approval queue | ✓ Phase 6 |
| 22 | Append-only audit log | ✓ Phase 6 |
| 48 | Shadow mode that stores a draft and does not send | ✓ Phase 6 |

---

## What is not in this phase

- No API endpoints exposing the approval queue, audit log, or drafts (Phase 7).
- No worker that drives the pipeline end-to-end (Phase 8).
- No live send path (Phase 8 worker; `store_draft` never sends).
