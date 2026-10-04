# Phase 7 Release Notes

**Product:** Buro Assistant
**Release tag:** phase-7
**Date:** 2026-10-03
**Checklist rows closed:** 23, 24

---

## Summary

Phase 7 delivers the operator desk: a set of read-only, tenant-scoped API endpoints that expose the inbound message queue, triage decisions, reply drafts, the human approval queue, the audit log, and the active task list. A dashboard endpoint (row 23) aggregates state counts across all message and draft tables. Department queues (row 24) filter messages by subject keyword so operators can view work by department. All endpoints are auth-gated and tenant-isolated.

---

## What was built

### `app/web/desk.py` — operator desk router

Mounted at `/desk` by `app/main.py` via `app.include_router()`.

All imports from `app.main` are deferred to function bodies to avoid the circular-import that would arise from `app.main` importing this module at module-load time — the same pattern already used by the `/ingest` route.

| Endpoint | Purpose | Row |
|---|---|---|
| `GET /desk/dashboard` | State counts across messages, drafts, and the held queue | 23 |
| `GET /desk/queue` | Inbound messages, newest first; optional `?state=` filter | — |
| `GET /desk/queue/{department}` | Messages whose `subject_normalized` contains the department keyword | 24 |
| `GET /desk/decisions` | Triage/supervisor decision rows, newest first | — |
| `GET /desk/drafts` | Draft rows, newest first; optional `?state=` filter | — |
| `GET /desk/approval` | Pending approval queue entries | — |
| `GET /desk/audit` | Recent audit log entries (default limit 100) | — |
| `GET /desk/tasks` | Active task list for the authenticated user | — |

#### Row 23 — dashboard counts (`GET /desk/dashboard`)

Returns:
```json
{
  "tenant_id": 1,
  "messages": {
    "received": 12,
    "classified": 8,
    "quarantine": 1,
    "duplicate": 2,
    "failed": 0
  },
  "drafts": {
    "drafted": 5,
    "sent": 3,
    "approved": 2,
    "rejected": 1
  },
  "held": 2
}
```
- `received` = messages in state `new`
- `classified` = messages in state `classified`
- `drafted` = drafts in state `draft` + `shadow` (both awaiting action)
- `held` = pending `approval_queue` entries

#### Row 24 — department queue (`GET /desk/queue/{department}`)

Filters `messages` by `func.lower(subject_normalized).contains(department_keyword)`. Case-insensitive, tenant-scoped. Returns the same message shape as `/desk/queue` plus the `department` echo field. The department keyword is a free-form string (e.g. `invoice`, `support`, `hr`).

---

## Migration

No new migration required — Phase 7 adds only API endpoints on top of the tables created in Phases 3–6.

---

## Tests

`tests/test_desk.py` — **29 tests** across 9 test classes:

| Class | Tests | Coverage |
|---|---|---|
| `TestDeskAuthRequired` | 1 | All 8 endpoints reject unauthenticated requests |
| `TestDashboard` | 8 | Row 23 — counts, tenant isolation, held queue |
| `TestDepartmentQueue` | 6 | Row 24 — keyword filter, case-insensitivity, tenant isolation, response shape |
| `TestDeskInbound` | 2 | Queue state filter |
| `TestDeskDecisions` | 1 | Decisions endpoint smoke test |
| `TestDeskDrafts` | 3 | Draft listing and state filter |
| `TestDeskApproval` | 2 | Approval queue view |
| `TestDeskAudit` | 2 | Audit log view |
| `TestDeskTasks` | 3 | Task list, active-only filter |

Full suite: **239 tests, 0 failures**.

---

## Capability matrix delta

| Row | Capability | Status |
|---|---|---|
| 23 | Dashboard counts: received, classified, drafted, sent, held, failed | ✓ Phase 7 |
| 24 | Department queues | ✓ Phase 7 |

---

## What is not in this phase

- No write endpoints on the desk (Phase 8 / Phase 9).
- No live message polling loop (Phase 8 worker).
- No cost/token fields on decisions (Phase 8).
- No department assignment stored in the `decisions` table from a live triage run (Phase 8 wires agents into the worker pipeline).
