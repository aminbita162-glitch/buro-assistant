# Buro Assistant — Install, Backup, Restore, and Threat Model

**Product:** Buro Assistant
**Version:** 1.0.0 → follow-up Phase 4
**Date:** 2026-10-06
**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab

---

## Install

### Prerequisites

| Requirement | Minimum version |
|---|---|
| Python | 3.9 |
| PostgreSQL | 14 |
| pip | 22 |

### Steps

1. **Clone the repository.**
   ```bash
   git clone <repo-url>
   cd buro-assistant
   ```

2. **Install Python dependencies.**
   ```bash
   pip install -r requirements.txt
   ```

3. **Copy the environment template and fill in values.**
   ```bash
   cp .env.example .env
   # Edit .env — set DATABASE_URL, OPENAI_API_KEY, ALLOWED_ORIGINS
   ```

   **Database region (operator-controlled):** Set `DATABASE_URL` to point to
   a database in the region you require.  Buro Assistant does not operate a
   hosted service and does not enforce or verify data residency automatically.
   If you must keep data in the EU (e.g. for GDPR compliance), host the
   database in an EU region.  This is your responsibility as the operator.
   See `docs/PRIVACY_DATA_MAP.md` for the full data map.

4. **Run database migrations.**
   Migrations run automatically on first startup.  To run them manually:
   ```bash
   alembic upgrade head
   ```

5. **Seed the demo tenant (optional).**
   ```bash
   python -m app.domain.seed
   ```
   For an isolated sandbox tenant:
   ```bash
   python -m app.domain.sandbox
   ```

6. **Start the server.**
   ```bash
   ./run.sh
   ```
   The operator UI is served at `http://localhost:8000`.

### Starting without a paid model key

The app starts and serves all routes without `OPENAI_API_KEY`.  Only the three
AI-assisted routes (`POST /assistant`, `POST /analyze`, `POST /tasks/{id}/ai-update`)
require a live key.  All other routes — ingest, desk, health, auth, policy, agents
via `FakeModel` — work without one.

To run agents in tests without a key, pass a `FakeModel` instance to `triage()`.
The fake model is in `app/agents/fake_model.py`.

---

## Backup

### What to back up

| Data | Location | Priority |
|---|---|---|
| PostgreSQL database | `DATABASE_URL` target | Critical |
| `.env` (secrets) | Operator's secret store | Critical |
| `migrations/` directory | Git repository | Covered by version control |
| `schemas/` directory | Git repository | Covered by version control |

### PostgreSQL backup

```bash
pg_dump "$DATABASE_URL" --format=custom --file=buro-backup-$(date +%Y%m%d).dump
```

Schedule this command as a cron job.  Retain at least 7 daily backups and 4 weekly backups.

### Verify the backup

```bash
pg_restore --list buro-backup-<date>.dump | head -20
```

---

## Restore

1. **Create a new empty database** (if restoring to a fresh instance).
   ```bash
   createdb buro_restore
   ```

2. **Restore the dump.**
   ```bash
   pg_restore --dbname=buro_restore --no-owner buro-backup-<date>.dump
   ```

3. **Update `DATABASE_URL`** in `.env` to point to the restored database.

4. **Run migrations** to apply any schema changes that occurred after the backup.
   ```bash
   alembic upgrade head
   ```

5. **Start the server.**
   ```bash
   ./run.sh
   ```

6. **Verify health.**
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/ready
   ```

---

## Threat model

> The full STRIDE threat model is in [`docs/THREAT_MODEL.md`](THREAT_MODEL.md).
> The section below is a summary retained for quick reference.

### Assets

| Asset | Sensitivity |
|---|---|
| Tenant mail content | High — may contain PII, contractual data |
| User passwords | High — Argon2id hashed; never stored in plaintext |
| Session tokens | High — stored as SHA-256 hash; raw token in bearer header only |
| API keys | High — stored as SHA-256 hash; raw key shown once |
| Webhook signing secrets | High — stored as SHA-256 hash; raw secret shown once |
| Database connection string | Critical — grants full data access |
| OpenAI API key | High — grants billing access |

### Threats and mitigations

| Threat | Mitigation |
|---|---|
| Credential theft via source control | `.env` is gitignored; `.env.example` contains no real values; DIRECTIVE.txt §2 forbids committing secrets |
| Password brute-force | Argon2id with cost factor; rate limit on `/auth/login` (10/minute) and `/auth/signup` (5/minute) |
| Session hijacking | Session token stored as SHA-256 hash in DB; raw token only in `Authorization: Bearer` header; 24-hour TTL with rotation on each login |
| API key leakage | Key stored as SHA-256 hash; `buro_` prefix makes keys identifiable in logs; scoped (read/ingest/admin) |
| Webhook replay attack | HMAC-SHA256 signature on every payload; receiver must verify `X-Buro-Signature` header using `verify_signature()` |
| Cross-tenant data access | `tenant_id` FK on every business table; all queries filtered by authenticated user's `tenant_id`; Phase 3 isolation tests enforce this |
| XSS via task or message fields | All task and message fields rendered with `textContent`, not `innerHTML` (Phase 2, row 38) |
| Clickjacking | `X-Frame-Options: DENY` and `frame-ancestors 'none'` in CSP (Phase 2, row 36) |
| CORS abuse | `ALLOWED_ORIGINS` from environment only; default is empty (deny all cross-origin) |
| PII leakage to model | Redaction layer removes emails, phones, cards, IBANs, SSNs from subject and body before any model call (Phase 5, row 13) |
| Model output injection (forbidden content) | Amilos checks reply body for invented prices, legal promises, and literal dates; raises `ValueError` on a hit |
| Attachment malware | Disallowed MIME types set `attachment_state = quarantine`; quarantined messages never reach the reply pipeline |
| Quota exhaustion | Per-tenant daily token quota; `QuotaExceeded` raised before model call; `BackpressureError` when queue cap is reached |
| Raw SQL exposure | `safe_db_error_message()` wraps all DB exceptions; raw SQL strings are never returned to clients |
| Schema drift | Alembic manages all migrations; `app/main.py` never calls `create_all` or `ALTER TABLE` at runtime |

### Out of scope for 1.0.0

- Transport encryption (TLS termination is expected at the reverse proxy / load balancer layer).
- Multi-region replication.
- SOC 2 / ISO 27001 certification.
- Commercial sale and hosted service — these are not done (see `docs/PRIVACY_DATA_MAP.md`).

---

*This document is part of the Buro Assistant 1.0.0 release. Owner: Amin Azimi, Azimi Innovation Lab.*
