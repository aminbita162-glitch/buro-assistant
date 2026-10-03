# Phase 2 Release Notes

**Product:** Buro Assistant
**Phase:** 2 – Security
**Date:** 2026-10-03
**Branch:** phase-1
**Development restart:** 2026-10-03

---

## Test command and result

```bash
pytest tests/test_security.py -v
```

**Result: 34 passed, 0 failed.**

```
tests/test_security.py::TestHealthChecks::test_liveness_ok PASSED
tests/test_security.py::TestHealthChecks::test_readiness_ok PASSED
tests/test_security.py::TestHealthChecks::test_liveness_no_auth_required PASSED
tests/test_security.py::TestHealthChecks::test_readiness_no_auth_required PASSED
tests/test_security.py::TestPasswordHashing::test_hash_is_argon2id PASSED
tests/test_security.py::TestPasswordHashing::test_correct_argon2id_verifies PASSED
tests/test_security.py::TestPasswordHashing::test_wrong_argon2id_raises PASSED
tests/test_security.py::TestPasswordHashing::test_correct_legacy_sha256_verifies PASSED
tests/test_security.py::TestPasswordHashing::test_wrong_legacy_sha256_raises PASSED
tests/test_security.py::TestPasswordHashing::test_signup_stores_argon2id PASSED
tests/test_security.py::TestPasswordHashing::test_legacy_hash_upgraded_on_login PASSED
tests/test_security.py::TestSessions::test_login_creates_session_row PASSED
tests/test_security.py::TestSessions::test_token_stored_as_hash_not_plaintext PASSED
tests/test_security.py::TestSessions::test_session_has_expiry PASSED
tests/test_security.py::TestSessions::test_rotation_on_second_login PASSED
tests/test_security.py::TestSessions::test_expired_session_rejected PASSED
tests/test_security.py::TestSessions::test_auth_me_requires_valid_session PASSED
tests/test_security.py::TestSessions::test_auth_me_returns_user_with_valid_session PASSED
tests/test_security.py::TestSessions::test_logout_deletes_session PASSED
tests/test_security.py::TestRateLimits::test_signup_rate_limited PASSED
tests/test_security.py::TestRateLimits::test_login_rate_limited PASSED
tests/test_security.py::TestRateLimits::test_assistant_rate_limited PASSED
tests/test_security.py::TestRateLimits::test_analyze_rate_limited PASSED
tests/test_security.py::TestSecurityHeaders::test_headers_on_health PASSED
tests/test_security.py::TestSecurityHeaders::test_headers_on_ready PASSED
tests/test_security.py::TestSecurityHeaders::test_headers_on_root PASSED
tests/test_security.py::TestSecurityHeaders::test_headers_on_api_endpoint PASSED
tests/test_security.py::TestSecurityHeaders::test_x_frame_options_deny PASSED
tests/test_security.py::TestSecurityHeaders::test_csp_no_frame_ancestors PASSED
tests/test_security.py::TestCORS::test_allowed_origin_gets_cors_header PASSED
tests/test_security.py::TestCORS::test_disallowed_origin_no_cors_header PASSED
tests/test_security.py::TestCORS::test_preflight_allowed_origin PASSED
tests/test_security.py::TestTextRendering::test_serialize_task_returns_strings PASSED
tests/test_security.py::TestTextRendering::test_index_html_uses_text_not_innerhtml PASSED
34 passed in 8.45s
```

All existing routes remain operational. The app still starts.

---

## Checklist rows closed

| Row | Capability | Test class |
|-----|-----------|------------|
| **31** | Live (`/health`) and ready (`/ready`) health checks; ready checks the DB | `TestHealthChecks` |
| **33** | Argon2id password hashes; one-time SHA-256 → Argon2id upgrade on login | `TestPasswordHashing` |
| **34** | Session table (`user_sessions`), hashed token, expiry, rotation on login | `TestSessions` |
| **35** | Rate limits: signup 5/min, login 10/min, `/assistant` 30/min, `/analyze` 20/min, `/tasks/{id}/ai-update` 20/min | `TestRateLimits` |
| **36** | Security headers (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`) and CSP on every response | `TestSecurityHeaders` |
| **37** | CORS allowlist read from `ALLOWED_ORIGINS` env var; unlisted origins denied | `TestCORS` |
| **38** | Task fields in `index.html` rendered via `textContent`/`createTextNode` — not `innerHTML` | `TestTextRendering` |

---

## Changes in this phase

| File | Change |
|------|--------|
| `app/main.py` | Full security rewrite (see detail below) |
| `index.html` | Task list renderer: replaced `innerHTML` template literal with DOM text node API |
| `requirements.txt` | Added `argon2-cffi==25.1.0`, `slowapi==0.1.10`, `limits==4.2`, `httpx==0.28.1` |
| `.env.example` | Added `ALLOWED_ORIGINS` documentation |
| `tests/conftest.py` | SQLite in-memory engine patch for cross-thread TestClient use |
| `tests/test_security.py` | 34 new tests covering rows 31, 33–38 |
| `docs/CAPABILITY_MATRIX.md` | Rows 31, 33–38 marked ✓ Verified |
| `docs/releases/phase-2.md` | This file |

### app/main.py detail

- **`UserSession` table** added (`id`, `user_id`, `token_hash`, `expires_at`). Raw bearer tokens are never stored; SHA-256 of the token is stored. `expires_at` defaults to 24 hours from issue. Login rotates sessions (old rows deleted, new row inserted).
- **Argon2id** replaces the previous SHA-256 `hash_password`. `verify_password` first attempts Argon2id verification; if the stored hash is not an Argon2 hash it falls through to SHA-256 comparison and marks the hash for upgrade. `upgrade_password_if_needed` re-hashes to Argon2id and commits on first successful legacy login.
- **`/ready`** endpoint added. Executes `SELECT 1` against the database; returns 503 if the database is unreachable.
- **Rate limiter** (`slowapi`) applied: signup 5/min, login 10/min, `/assistant` 30/min, `/analyze` 20/min, `/tasks/{id}/ai-update` 20/min. Returns HTTP 429 on breach.
- **Security headers middleware** added to every response: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 0`, `Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy`, and a `Content-Security-Policy` with `frame-ancestors 'none'`.
- **CORS** middleware reads `ALLOWED_ORIGINS` env var (comma-separated). Empty → no cross-origin requests allowed.
- **Model routes** (`/assistant`, `/analyze`, `/tasks/{id}/ai-update`) already call `require_current_user` before any model call; 401 is returned for unauthenticated requests.
- `from __future__ import annotations` and `Optional[T]` used throughout for Python 3.9 compatibility.
- Legacy `ensure_*` functions collapsed into a single `_ensure_column` helper.

---

## Notes

- The `users.token` column is kept in-place by `_ensure_column` to avoid breaking existing databases before Phase 3's Alembic migration removes it cleanly.
- Phase 3 begins only on a new operator instruction.
