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

Current build (Phase 1): authentication, task management, and AI-assisted task operations are operational. Later phases add tenancy, ingest, agents, policy, workers, and the operator desk.

---

## Agents

**Triage** — Accepts a normalized message and the tenant rule pack. Returns JSON valid against `schemas/triage_decision.json`. Does not send mail and does not write billing. If a rule hits, the model client is not called.

**Reply** — Accepts the triage decision and an approved template. Returns JSON valid against `schemas/reply_draft.json`. Cannot invent a price, a legal promise, or a date that is not in the template variables.

**Supervisor** — Accepts an exception. Returns JSON valid against `schemas/supervisor_decision.json`. Allowed values: `hold`, `request_human`, `reject`, `reroute`. Cannot send or delete.

Model output that fails schema validation becomes a supervisor exception and is not written as a decision.

---

## Architecture

```
app/
  api/        routes and dependencies
  domain/     tenants, users, sessions, messages, tasks
  ingest/     providers, normalization, idempotency
  agents/     triage.py, reply.py, supervisor.py
  policy/     rules, templates, send decision
  workers/    queue, lanes, dead letter
  web/        operator UI
migrations/   Alembic only
schemas/      JSON schemas for the three agent outputs
tests/        unit, contract, tenant isolation, golden messages
docs/         install, backup, threat model, runbook, capability matrix, releases
```

The entry point is `app/main.py`. The server is started with `run.sh`. Schema changes are managed by Alembic migrations only; no schema change occurs on import (from Phase 3 onwards).

---

## Verified in this release

Phase 1 closes checklist rows: 39 (capability matrix present).

No throughput numbers are published until the benchmark in `docs/BENCHMARK.md` is written and run (Phase 8).

---

## Designed for, not yet measured

- Many tenants and horizontal workers
- Horizontal queue workers with priority lanes
- Per-tenant daily quota and backpressure

These capabilities are designed into the architecture. Verified numbers will appear in `docs/BENCHMARK.md` after the Phase 8 benchmark is run.

---

## Quick start

```bash
cp .env.example .env
# Edit .env and set DATABASE_URL and OPENAI_API_KEY
pip install -r requirements.txt
./run.sh
```

The operator UI is served at `http://localhost:8000`.

---

## Configuration

All configuration is read from environment variables. See `.env.example` for the full list. Secrets must never be committed to the repository.

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `OPENAI_API_KEY` | OpenAI API key for model routes |
| `PORT` | Bind port (default `8000`) |

---

## Tests

```bash
pytest tests/
```

No tests exist yet. The test suite is added in Phases 2–10.

---

## Release

Development restart: 2026-10-03.
Current version: Phase 1.
Target release: 1.0.0.

Release notes are in `docs/releases/`.

---

## Author

Buro Assistant is an office mail desk built by Amin Azimi, AI Architect, Azimi Innovation Lab. It receives tenant mail, classifies it, drafts a short receipt, and places the work on the right queue so an operator can see what arrived, what was decided, and what was sent.
