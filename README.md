# Buro Assistant

## Contents

- [What it does](#what-it-does)
- [Agents](#agents)
- [Architecture](#architecture)
- [Verified in this release](#verified-in-this-release)
- [Designed for, not yet measured](#designed-for-not-yet-measured)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Tests](#tests)
- [Release](#release)
- [Author](#author)

---

## What it does

Buro Assistant is a multi-tenant office mail desk. It receives messages through a configured mailbox adapter, stores them once, classifies them with fixed rules before any model call, drafts a short receipt from an approved template, sends that receipt only when the tenant policy allows, and shows the operator the inbound queue, decisions, drafts, sent mail, failures, cost, and audit trail. Tasks extracted from a message stay attached to that message and that tenant.

---

## Agents

**Amin (triage)** — Accepts a normalized message and the tenant rule pack. Returns JSON valid against `schemas/triage_decision.json`. Does not send mail and does not write billing. If a rule hits, the model client is not called. Confidence below threshold routes to Leila.

**Amilos (reply)** — Accepts the triage decision and an approved template. Returns JSON valid against `schemas/reply_draft.json`. Cannot invent a price, a legal promise, or a date that is not in the template variables.

**Leila (supervisor)** — Accepts an exception. Returns JSON valid against `schemas/supervisor_decision.json`. Allowed actions: `hold`, `request_human`, `reject`, `reroute`. Cannot send or delete.

Model output that fails schema validation becomes a supervisor exception and is not written as a decision. The same tenant rules and the same normalized message, with the model disabled, produce the same decision hash.

---

## Architecture

```
app/
  agents/     Amin (triage), Amilos (reply), Leila (supervisor)
  api/        routes and dependencies
  domain/     tenants, users, sessions, tasks, usage, export, retention, sandbox, API keys, webhooks
  ingest/     providers (IMAP, fake), normalization, idempotency
  policy/     templates, send decision, SLA, calendar, approval queue, audit log, shadow mode
  workers/    queue (priority lanes), quota, backpressure, dead letter, traces, cost
  web/        operator desk router (/desk/*)
migrations/   Alembic versions 0001–0006
schemas/      triage_decision.json, reply_draft.json, supervisor_decision.json
tests/        unit, contract, tenant isolation, golden messages, release checks
docs/         INSTALL.md, RUNBOOK.md, BENCHMARK.md, CAPABILITY_MATRIX.md, releases/
```

The entry point is `app/main.py`. The server is started with `run.sh`. Schema changes are managed by Alembic migrations only; no schema change occurs on import.

---

## Verified in this release

All fifty capability checklist rows are closed in release 1.0.0. See `docs/CAPABILITY_MATRIX.md` for the full matrix and `docs/releases/1.0.0.md` for the row-by-row test pointer list.

Key verified capabilities:

- Tenant isolation on every business table, with cross-tenant rejection tests
- Idempotency, raw store, duplicate detection, attachment quarantine
- Three-agent pipeline: rule engine, redaction, language detection, urgency lexicon, confidence threshold, schema validation, deterministic decision hash
- Template registry with forbidden phrases, receipt template, auto-reply gate, shadow mode
- Business-hours calendar, SLA clock, human approval queue, append-only audit log
- Operator desk: dashboard counts, department queues, decisions, drafts, audit
- Priority-lane work queue, backpressure, dead-letter replay, traces, per-tenant quota
- Usage events, tenant export, retention (default 180 days), sandbox seed
- Scoped API keys (SHA-256 hash), signed outbound webhooks (HMAC-SHA256)

---

## Designed for, not yet measured

- Many tenants and horizontal workers
- Horizontal queue workers with priority lanes
- Per-tenant daily quota and backpressure

These capabilities are designed into the architecture. Measured numbers are in `docs/BENCHMARK.md`.

---

## Quick start

```bash
cp .env.example .env
# Edit .env — set DATABASE_URL and OPENAI_API_KEY
pip install -r requirements.txt
./run.sh
```

The app starts without an `OPENAI_API_KEY`. Only `POST /assistant`, `POST /analyze`, and `POST /tasks/{id}/ai-update` require a live key. All other routes — ingest, desk, health, auth, policy, and agents via `FakeModel` — work without one.

The operator UI is served at `http://localhost:8000`.

To seed a sandbox tenant:

```bash
python -m app.domain.sandbox
```

---

## Configuration

All configuration is read from environment variables. See `.env.example` for the full list. Secrets must never be committed to the repository.

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `OPENAI_API_KEY` | OpenAI API key (model routes only) |
| `PORT` | Bind port (default `8000`) |
| `ALLOWED_ORIGINS` | Comma-separated CORS origin allowlist |
| `IMAP_HOST` | IMAP server hostname |
| `IMAP_PORT` | IMAP server port (default `993`) |
| `IMAP_USER` | IMAP username |
| `IMAP_PASSWORD` | IMAP password |

---

## Tests

```bash
python3 -m pytest tests/ -v
```

329 tests, 0 failures (release 1.0.0).

Test files and what they cover:

| File | Rows covered |
|---|---|
| `tests/test_security.py` | 31, 33, 34, 35, 36, 37, 38 |
| `tests/test_tenancy.py` | 1, 2, 32 |
| `tests/test_ingest.py` | 3, 4, 5, 6, 9, 12 |
| `tests/test_agents.py` | 7, 8, 10, 11, 13, 14, 15, 46 |
| `tests/test_contracts.py` | 47 |
| `tests/test_policy.py` | 16, 17, 18, 19, 20, 21, 22, 48 |
| `tests/test_desk.py` | 23, 24 |
| `tests/test_workers.py` | 25, 26, 27, 28, 29, 30 |
| `tests/test_commercial.py` | 40, 41, 42, 43, 44, 45 |
| `tests/test_release.py` | 39, 49, 50 |

---

## Release

Development restart: 2026-10-03.
Current version: 1.0.0.

Release notes are in `docs/releases/`. The full 50-row checklist with test pointers is in `docs/releases/1.0.0.md`.

Install, backup, restore, and threat model: `docs/INSTALL.md`.
Runbook (incident, quota breach, bad template): `docs/RUNBOOK.md`.
Benchmark: `docs/BENCHMARK.md`.

---

## Author

Buro Assistant is an office mail desk built by Amin Azimi, AI Architect, Azimi Innovation Lab. It receives tenant mail, classifies it, drafts a short receipt, and places the work on the right queue so an operator can see what arrived, what was decided, and what was sent.
