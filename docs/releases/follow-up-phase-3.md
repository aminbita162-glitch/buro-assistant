# Follow-up Phase 3 — Operator desk that a buyer can use

**Product:** Buro Assistant  
**Contract:** Follow-up contract (development restart 2026-10-04)  
**Phase:** 3 – Operator desk that a buyer can use  
**Date:** 2026-10-06  
**Branch:** main  

---

## What this phase does

Turns the existing `index.html` into a usable operator desk and wires all
supporting back-end pieces:

```
index.html  →  Operator Desk
  ├── Login screen (no innerHTML for message fields)
  ├── Dashboard   – message counts, draft counts, held-for-approval count
  ├── Inbound     – paginated table, state + department filters
  ├── Decisions   – decision log table
  ├── Drafts      – draft table with state filter
  ├── Approval    – pending entries, Approve / Reject buttons (POST)
  ├── Audit       – audit log table
  ├── Quota       – today's token quota and usage
  └── Cost        – usage event summary and list

Back-end additions (app/web/desk.py):
  POST /desk/approval/{id}/approve  →  policy.approval.resolve("approved")
  POST /desk/approval/{id}/reject   →  policy.approval.resolve("rejected")
  GET  /desk/quota                  →  workers.quota.get_or_create_quota()
  GET  /desk/cost                   →  domain.usage.events_for_tenant()

Department now stored:
  messages.department  – new nullable column (migration 0008).
  /desk/queue/{dept}   – matches stored department field first, then falls
                         back to subject-keyword contains for old rows.
  _ser_message()       – now includes "department" in every message dict.
```

---

## Test command

```bash
python3 -m pytest tests/ -q
```

**Result:** 413 passed, 0 failed, 0 errors.

Previous baseline: 396 tests (Phase 2).  
New tests this phase: 17 (in `tests/test_desk.py`).

---

## Requirements satisfied

| Requirement | Satisfied | Notes |
|---|---|---|
| Existing index page becomes the desk | yes | Full rewrite; task-manager UI replaced by operator desk |
| Desk sections: inbound, decision, draft, approval, audit, quota, cost | yes | Eight sections in sidebar; all load from existing `/desk/*` endpoints |
| No innerHTML for message fields | yes | All user-supplied data rendered with `el()` helper using `textContent` only |
| Department is a stored field, not only a subject keyword | yes | `messages.department` column added (migration 0008); ORM updated; serialiser updated |
| Approve and reject call the existing queue | yes | `POST /desk/approval/{id}/approve` and `.../reject` call `policy.approval.resolve()` |
| Tenant scoped | yes | All new endpoints use `_open_db_and_auth()` and filter by `user.tenant_id` |
| App stays runnable | yes | `app/main.py` and all existing routes unchanged |

---

## Files changed

| File | Change |
|---|---|
| `migrations/versions/0008_message_department.py` | **New.** Adds `messages.department` column and index. |
| `app/ingest/models.py` | Added `department = Column(String, nullable=True, index=True)` to `Message`. |
| `app/web/desk.py` | Added `approve_entry`, `reject_entry`, `desk_quota`, `desk_cost` endpoints; updated `department_queue` to use stored field with subject-keyword fallback; updated `_ser_message` to include `department`; updated module docstring. |
| `index.html` | **Rewritten.** Operator desk with login screen, sidebar navigation, eight sections, DOM-safe rendering (no innerHTML for data). |
| `tests/test_desk.py` | Added Phase 3 test classes: `TestApproveReject` (7 tests), `TestDeskQuota` (4 tests), `TestDeskCost` (3 tests), `TestDepartmentStoredField` (2 tests), `test_approve_reject_require_auth` (1 test); added `/desk/quota` and `/desk/cost` to `TestDeskAuthRequired`. |
| `docs/releases/follow-up-phase-3.md` | **New.** This file. |

---

## Known limitations (not failures)

- The `department` column is nullable. Existing rows and rows ingested without
  an explicit department remain `NULL`. The department-queue fallback (subject
  keyword) ensures those rows are still surfaced.
- The desk UI is a single-page static file served from the root of the API
  process. It talks to the same origin. If the API is on a different origin,
  set the `API` constant in `index.html` to the API base URL and ensure CORS
  is configured accordingly.
- `POST /desk/approval/{id}/approve` and `reject` record `resolved_by` as the
  operator's email. No audit log event is written for the resolution in this
  phase (audit-log integration is out of scope here; it would be a Phase 4+
  item).

---

## Test failures

None. All 413 tests pass.
