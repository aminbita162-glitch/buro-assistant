# Phase 3 Release Notes

**Product:** Buro Assistant
**Phase:** 3 – Tenancy and migrations
**Date:** 2026-10-03
**Branch:** main
**Development restart:** 2026-10-03

---

## Test command and result

```bash
pytest tests/ -v
```

**Result: 51 passed, 0 failed, 0 warnings.**

```
tests/test_security.py  – 34 passed   (Phase 2 rows, all still green)
tests/test_tenancy.py   – 17 passed   (Phase 3 rows 1, 2, 32)
51 passed in 10.31s
```

The app still starts. All existing routes remain operational.

---

## Checklist rows closed

| Row | Capability | Test class |
|-----|-----------|------------|
| **1** | `tenant_id` on every business table (`tenants`, `users`, `user_sessions`, `tasks`) | `TestTenantIdColumns` (8 tests) |
| **2** | Queries enforce cross-tenant isolation; 404 returned for any cross-tenant read, update, or delete via the API | `TestCrossTenantIsolation` (5 tests) |
| **32** | Alembic baseline migration `0001_baseline`; `Base.metadata.create_all` and `_ensure_column` removed from `app/main.py`; schema managed exclusively by Alembic | `TestAlembicMigration` (4 tests) |

---

## Changes in this phase

| File | Change |
|------|--------|
| `app/main.py` | Added `Tenant` model; added `tenant_id` FK to `User`, `UserSession`, `Task`; removed `_ensure_column` bootstrap and `Base.metadata.create_all`; replaced with `_run_migrations()` (Alembic startup call) |
| `app/domain/seed.py` | Demo seed command — creates `demo` tenant, demo user, sample task; idempotent |
| `alembic.ini` | Alembic configuration wired to `DATABASE_URL` env var |
| `migrations/env.py` | Alembic env — imports `app.main.Base`; supports offline and online modes |
| `migrations/script.py.mako` | Alembic migration template |
| `migrations/versions/0001_baseline.py` | Baseline migration: creates `tenants`, `users`, `user_sessions`, `tasks`; handles existing databases by detecting and adding missing columns |
| `requirements.txt` | Added `alembic==1.16.5` |
| `tests/conftest.py` | Added Alembic `engine_from_config` patch for SQLite cross-thread use |
| `tests/test_security.py` | Updated `_make_user`, `_make_legacy_user`, and inline User construction to supply `tenant_id` |
| `tests/test_tenancy.py` | 17 new tests covering rows 1, 2, 32 |
| `docs/CAPABILITY_MATRIX.md` | Rows 1, 2, 32 marked ✓ Verified |
| `docs/releases/phase-3.md` | This file |

### app/main.py detail

- **`Tenant` model** — `id`, `name` (unique), `slug` (unique, indexed).
- **`tenant_id` FK** added to `User`, `UserSession`, and `Task` pointing at `tenants.id`.
- **`get_active_tasks_query`** and **`get_active_tasks`** now require `tenant_id` and filter `Task.tenant_id == tenant_id` in addition to `user_id`.
- Every individual task fetch (GET, PUT, DELETE, ai-update, ai-delete) filters `Task.tenant_id == user.tenant_id`.
- **`create_session`** now takes `tenant_id` and stores it on `UserSession`.
- **`signup`** resolves or creates the `default` tenant for self-service registrations.
- **`_run_migrations()`** replaces the old `create_all` / `_ensure_column` bootstrap; called once at process startup; failures are logged but do not crash the process.

### Seed command

```bash
python -m app.domain.seed
```

Creates tenant slug=`demo`, user `demo@example.com` (password printed to stdout), and a sample task. Idempotent — safe to run multiple times.

---

## Notes

- The `users.token` legacy column is dropped in `0001_baseline` when migrating existing databases (`batch_alter_table` with `drop_column`).
- The baseline migration is idempotent for the `tasks`, `users`, and `user_sessions` tables: it detects existing tables and only adds missing columns.
- Phase 4 begins only on a new operator instruction.
