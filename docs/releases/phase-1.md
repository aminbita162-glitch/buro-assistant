# Phase 1 Release Notes

**Product:** Buro Assistant
**Phase:** 1 – Governance and layout
**Date:** 2026-10-03
**Branch:** phase-1
**Development restart:** 2026-10-03

---

## Test command

```bash
pytest tests/
```

**Result:** No tests exist yet. The test harness is in place (`tests/` package created). Tests are added from Phase 2 onwards.

The application starts and serves all existing routes:

```bash
./run.sh
# GET  /         → operator UI (index.html)
# GET  /health   → {"status": "ok"}
# POST /auth/signup
# POST /auth/login
# POST /auth/logout
# GET  /auth/me
# GET  /stats
# GET  /tasks
# GET  /tasks/search
# GET  /tasks/{id}
# DELETE /tasks/{id}
# PUT  /tasks/{id}
# POST /tasks/{id}/ai-update
# POST /tasks/ai-delete
# POST /assistant
# POST /analyze
```

---

## Checklist rows closed

| Row | Capability |
|-----|-----------|
| 39 | Capability matrix added (`docs/CAPABILITY_MATRIX.md`) with Designed for and Verified columns |

---

## Changes in this phase

| File | Change |
|------|--------|
| `LICENSE` | Added; text from DIRECTIVE.txt section 7 |
| `README.md` | Replaced; shape from DIRECTIVE.txt section 8 |
| `docs/CAPABILITY_MATRIX.md` | Added; all 50 checklist rows with Designed for and Verified columns |
| `docs/releases/phase-1.md` | This file |
| `.env.example` | Added; documents all environment variables |
| `requirements.txt` | Dependencies pinned to exact latest versions |
| `run.sh` | Added shebang, `set -e`, and `PORT` default |
| `app/__init__.py` | Added; marks `app` as a Python package |
| `app/api/__init__.py` | Stub; implementation in Phase 2+ |
| `app/domain/__init__.py` | Stub; implementation in Phase 3+ |
| `app/ingest/__init__.py` | Stub; implementation in Phase 4+ |
| `app/agents/__init__.py` | Stub; implementation in Phase 5+ |
| `app/policy/__init__.py` | Stub; implementation in Phase 6+ |
| `app/workers/__init__.py` | Stub; implementation in Phase 8+ |
| `app/web/__init__.py` | Stub; implementation in Phase 7+ |
| `migrations/` | Directory created; Alembic baseline in Phase 3 |
| `schemas/` | Directory created; agent schemas in Phase 5 |
| `tests/__init__.py` | Test package created; tests added from Phase 2 |

All existing routes remain operational. `app/main.py` is unchanged.

---

## Notes

- No business logic was moved or removed. The single-file `app/main.py` remains the entry point and all routes it provides continue to work.
- The sub-package stubs (`app/api/`, `app/domain/`, etc.) are empty. Code will be moved into them phase by phase once tested replacements exist.
- Phase 2 begins only on a new operator instruction.
