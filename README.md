# Buro Assistant

**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab
**Current version:** 1.0.0 (follow-up Phase 8 closed)
**License:** see `LICENSE`

---

## At a glance

Buro Assistant is a self-hosted, multi-tenant office mail desk.
It polls a mailbox, classifies each message with rules before any model call,
drafts a short receipt from an approved template, and routes the work to an
operator queue. The operator approves, rejects, or lets the system send.
Everything is tenant-scoped. Nothing is invented — no prices, no dates, no
facts beyond what the template allows.

**Five things a buyer checks first:**

1. **Rule before model** — a rule match produces zero model tokens.
2. **Redaction before model** — PII fields are stripped before the prompt is built.
3. **Shadow mode** — drafts are stored and not sent unless the operator enables auto-reply.
4. **Append-only audit log** — every decision is recorded and cannot be overwritten.
5. **Self-hosted** — the database region is operator-controlled. There is no hosted service.

---

## Contents

- [What it does](#what-it-does)
- [Architecture diagram](#architecture-diagram)
- [Agents](#agents)
- [File map](#file-map)
- [Designed versus verified](#designed-versus-verified)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Tests](#tests)
- [Roadmap](#roadmap)
- [What is not claimed](#what-is-not-claimed)
- [External links](#external-links)
- [Author](#author)

---

## What it does

Buro Assistant is a multi-tenant office mail desk. It receives messages through a
configured mailbox adapter, stores them once with an idempotency key, classifies
them with fixed rules before any model call, drafts a short receipt from an approved
template, and routes the result to the operator queue. The operator sees the inbound
queue, decisions, drafts, sent mail, failures, cost, and an append-only audit trail.
Tasks extracted from a message stay scoped to that message and that tenant.

A rule match never calls a model. Attachment bytes are never sent to a model.
Redaction runs before every model call. Shadow mode stores a draft and does not send.

---

## Architecture diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  External mail server (operator-hosted)                         │
│  IMAP / SMTP                                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  app/ingest/                                                    │
│  IMAPProvider  or  FakeProvider (no mailbox env set)            │
│  → normalize → idempotency check → raw store                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  app/agents/                                                    │
│  Amin (triage)                                                  │
│    rule engine → rule hit? ──yes──→ decision (no model call)   │
│                         │                                       │
│                         no                                      │
│                         ▼                                       │
│    redact → language detect → urgency → model call             │
│    schema validate → decision hash                             │
│                         │                                       │
│    confidence < threshold ──→ Leila (supervisor)               │
│                         │                                       │
│    confidence ≥ threshold ──→ Amilos (reply)                   │
│    template registry → forbidden-phrase check → draft          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  app/policy/                                                    │
│  tenant send policy                                             │
│    shadow mode ──→ draft stored, not sent                      │
│    approval required ──→ human approval queue                  │
│    auto-reply on ──→ outbound delivery + signed webhook        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  app/workers/                                                   │
│  priority-lane queue  •  quota  •  backpressure                │
│  dead-letter replay   •  traces  •  cost events                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  app/web/  (operator desk at /desk/*)                           │
│  dashboard  •  inbound  •  decisions  •  drafts                │
│  approval   •  audit    •  quota/cost •  privacy               │
└─────────────────────────────────────────────────────────────────┘
```

Full diagram source: [`docs/diagrams/architecture.txt`](docs/diagrams/architecture.txt)

---

## Agents

**Amin (triage)** — Accepts a normalized message and the tenant rule pack.
Returns JSON valid against [`schemas/triage_decision.json`](schemas/triage_decision.json).
Does not send mail and does not write billing. If a rule hits, the model client
is not called. Confidence below threshold routes to Leila.

**Amilos (reply)** — Accepts the triage decision and an approved template.
Returns JSON valid against [`schemas/reply_draft.json`](schemas/reply_draft.json).
Cannot invent a price, a legal promise, or a date that is not in the template variables.

**Leila (supervisor)** — Accepts an exception. Returns JSON valid against
[`schemas/supervisor_decision.json`](schemas/supervisor_decision.json).
Allowed actions: `hold`, `request_human`, `reject`, `reroute`. Cannot send or delete.

Model output that fails schema validation becomes a supervisor exception and is not
written as a decision. The same tenant rules and the same normalized message, with
the model disabled, produce the same decision hash.

---

## File map

| Package | Responsibility |
|---|---|
| `app/agents/` | Amin triage, Amilos reply, Leila supervisor, redaction, rules, urgency, language, decision hash |
| `app/api/` | FastAPI routes, Bearer auth dependency, rate limiting |
| `app/domain/` | Tenants, users, sessions, tasks, usage events, export, retention, sandbox seed, API keys, webhooks, buyer features (15 Section B options) |
| `app/ingest/` | IMAP provider, fake provider, normalization, idempotency, raw store |
| `app/policy/` | Template registry, send decision, SLA clock, business-hours calendar, approval queue, audit log, shadow mode |
| `app/workers/` | Priority-lane queue, per-tenant quota, backpressure, dead-letter queue, traces, cost events |
| `app/web/` | Operator desk router (`/desk/*`), dashboard, privacy endpoints |
| `app/main.py` | FastAPI application entry point, middleware, security headers |
| `migrations/` | Alembic versions 0001–0010 — all schema changes are here, never in import |
| `schemas/` | JSON schemas for triage_decision, reply_draft, supervisor_decision |
| `tests/` | Unit, contract, tenant isolation, golden messages, release checks |
| `docs/` | INSTALL, RUNBOOK, BENCHMARK, CAPABILITY_MATRIX, THREAT_MODEL, PRIVACY_DATA_MAP, COMPARISON, TEST_HOUSE, INSTALL_VIDEO_SCRIPT, releases, diagrams |

---

## Designed versus verified

| Capability | Designed for | Verified |
|---|---|---|
| Tenant isolation on every business table | ✓ | ✓ `tests/test_tenancy.py` |
| Cross-tenant rejection tests | ✓ | ✓ `tests/test_tenancy.py` |
| Idempotency key (provider message id + tenant) | ✓ | ✓ `tests/test_ingest.py` |
| Immutable raw message store | ✓ | ✓ `tests/test_ingest.py` |
| IMAP adapter, credentials from environment only | ✓ | ✓ `tests/test_ingest.py` |
| Provider-neutral normalized message | ✓ | ✓ `tests/test_ingest.py` |
| Tenant rule pack (domain, subject, department) | ✓ | ✓ `tests/test_agents.py` |
| Confidence threshold → Leila | ✓ | ✓ `tests/test_agents.py` |
| Duplicate detection by Message-Id and subject | ✓ | ✓ `tests/test_ingest.py` |
| Language detection | ✓ | ✓ `tests/test_agents.py` |
| Urgency lexicon with tenant overrides | ✓ | ✓ `tests/test_agents.py` |
| Attachment allowlist and quarantine | ✓ | ✓ `tests/test_ingest.py` |
| Redaction before model call | ✓ | ✓ `tests/test_agents.py` |
| Prompt and schema version on every decision | ✓ | ✓ `tests/test_agents.py` |
| Schema validation of agent output | ✓ | ✓ `tests/test_agents.py` |
| Template registry with forbidden phrases | ✓ | ✓ `tests/test_policy.py` |
| Auto-reply off unless tenant enables it | ✓ | ✓ `tests/test_policy.py` |
| Receipt template | ✓ | ✓ `tests/test_policy.py` |
| Business-hours calendar per tenant | ✓ | ✓ `tests/test_policy.py` |
| SLA clock from ingest time | ✓ | ✓ `tests/test_policy.py` |
| Human approval queue | ✓ | ✓ `tests/test_policy.py` |
| Append-only audit log | ✓ | ✓ `tests/test_policy.py` |
| Dashboard counts (received, classified, drafted, sent, held, failed) | ✓ | ✓ `tests/test_desk.py` |
| Department queues | ✓ | ✓ `tests/test_desk.py` |
| Cost and token fields on model calls | ✓ | ✓ `tests/test_workers.py` |
| Per-tenant daily token quota | ✓ | ✓ `tests/test_workers.py` |
| Backpressure when queue depth exceeds cap | ✓ | ✓ `tests/test_workers.py` |
| Dead-letter queue and replay | ✓ | ✓ `tests/test_workers.py` |
| Priority lanes (critical > high > medium > low) | ✓ | ✓ `tests/test_workers.py` |
| Traces around ingest, decide, draft, and send | ✓ | ✓ `tests/test_workers.py` |
| Live and ready health checks | ✓ | ✓ `tests/test_security.py` |
| Alembic migrations; no schema change on import | ✓ | ✓ `tests/test_security.py` |
| Argon2id password hashing | ✓ | ✓ `tests/test_security.py` |
| Session table, hashed token, expiry, rotation | ✓ | ✓ `tests/test_security.py` |
| Rate limits on signup, login, and model routes | ✓ | ✓ `tests/test_security.py` |
| Security headers and content security policy | ✓ | ✓ `tests/test_security.py` |
| CORS allowlist from environment | ✓ | ✓ `tests/test_security.py` |
| Message and task fields rendered as text, not HTML | ✓ | ✓ `tests/test_security.py` |
| Capability matrix | ✓ | ✓ `tests/test_release.py` |
| Usage events | ✓ | ✓ `tests/test_commercial.py` |
| Tenant data export | ✓ | ✓ `tests/test_commercial.py` |
| Retention (default 180 days) | ✓ | ✓ `tests/test_commercial.py` |
| Sandbox tenant seed command | ✓ | ✓ `tests/test_commercial.py` |
| Scoped API keys, stored hashed | ✓ | ✓ `tests/test_commercial.py` |
| Signed outbound webhooks (HMAC-SHA256) | ✓ | ✓ `tests/test_commercial.py` |
| Fifty golden messages with expected decisions | ✓ | ✓ `tests/test_agents.py` |
| Contract tests for three schemas | ✓ | ✓ `tests/test_contracts.py` |
| Shadow mode: draft stored, not sent | ✓ | ✓ `tests/test_policy.py` |
| Runbook | ✓ | ✓ `docs/RUNBOOK.md` |
| Install, backup, restore, threat model, release notes | ✓ | ✓ `docs/INSTALL.md` |
| Rule before model — zero tokens on rule hit | ✓ | ✓ `tests/test_token_policy.py` |
| Attachment bytes never sent to model | ✓ | ✓ `tests/test_token_policy.py` |
| Prompt body clipped to ≤ 200 chars | ✓ | ✓ `tests/test_token_policy.py` |
| Rule pack hash in every model prompt | ✓ | ✓ `tests/test_token_policy.py` |
| Triage cache for rule-hit inputs | ✓ | ✓ `tests/test_token_policy.py` |
| Live mailbox poll (IMAP worker loop) | ✓ | designed for — not yet externally measured |
| Many tenants, horizontal workers | ✓ | designed for — not yet externally measured |
| Per-tenant quota backpressure at scale | ✓ | designed for — not yet externally measured |

Full matrix: [`docs/CAPABILITY_MATRIX.md`](docs/CAPABILITY_MATRIX.md)

---

## Air-gap package (operator-hosted)

Buro Assistant ships a `docker-compose.yml` for fully operator-hosted deployment
with **no cloud model** and **no gateway secret**.

```bash
docker compose up
```

The compose file starts:

- A PostgreSQL 16 database (`db` service, data in a named volume).
- The application (`app` service) with `LOCAL_MODEL=1` so the cloud model
  client is never called — `FakeModel` is used for all agent calls.
- `OPENAI_API_KEY` and `GATEWAY_SECRET` are intentionally absent from the
  compose file. The fake payment adapter is used when `GATEWAY_SECRET` is not
  set. No secret value is embedded in any tracked file.

The operator desk is at `http://localhost:8000` once the stack is up.

> **This is operator-hosted.** Azimi Innovation Lab does not operate or monitor
> this stack. The operator controls the host, the database, and all credentials.

---

## Quick start

```bash
git clone <repo-url>
cd buro-assistant
cp .env.example .env
# Edit .env — set DATABASE_URL and OPENAI_API_KEY
pip install -r requirements.txt
./run.sh
```

The app starts without `OPENAI_API_KEY`. Only `POST /assistant`, `POST /analyze`,
and `POST /tasks/{id}/ai-update` require a live key. All other routes — ingest,
desk, health, auth, policy, and agents via `FakeModel` — work without one.

The operator desk is at `http://localhost:8000`.

To seed a sandbox tenant:

```bash
python -m app.domain.sandbox
```

To run migrations manually:

```bash
alembic upgrade head
```

---

## Configuration

All configuration is read from environment variables. See [`.env.example`](.env.example)
for the full list. Never commit `.env` to version control.

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

Database region is operator-controlled. Buro Assistant does not operate a hosted
service and does not enforce or verify data residency automatically. See
[`docs/INSTALL.md`](docs/INSTALL.md) and [`docs/PRIVACY_DATA_MAP.md`](docs/PRIVACY_DATA_MAP.md).

---

## Tests

```bash
python3 -m pytest tests/ -v
```

504 tests, 0 failures (follow-up Phase 7 baseline).

| File | Covers |
|---|---|
| `tests/test_security.py` | security headers, rate limits, health, migrations |
| `tests/test_tenancy.py` | tenant isolation, cross-tenant rejection |
| `tests/test_ingest.py` | idempotency, raw store, IMAP adapter, normalization |
| `tests/test_agents.py` | rule engine, triage, language, urgency, golden messages |
| `tests/test_contracts.py` | JSON schema contracts for three agent schemas |
| `tests/test_policy.py` | templates, send decision, SLA, approval queue, audit, shadow mode |
| `tests/test_desk.py` | dashboard counts, department queues |
| `tests/test_workers.py` | queue, quota, backpressure, dead letter, traces, cost |
| `tests/test_commercial.py` | usage, export, retention, sandbox, API keys, webhooks |
| `tests/test_release.py` | capability matrix, release surface |
| `tests/test_token_policy.py` | rule-hit zero tokens, attachment exclusion, triage cache, prompt shape |
| `tests/test_buyer_features.py` | all 15 Section B buyer options |
| `tests/test_differentiation.py` | all 15 Section C finish items |

---

## Commercial status

**Sale wiring is in progress.** The plan catalogue and subscription record are implemented. Payment collection is not yet wired; the payment boundary uses a fake adapter until a gateway secret is present in the environment. No card data is stored.

**Hosting is not offered.** Buro Assistant is self-hosted by the operator. Azimi Innovation Lab does not operate a hosted service, shared infrastructure, or a SaaS offering.

---

## Roadmap

Commercial contract phases:

| Phase | Title | Status |
|---|---|---|
| Commercial 1 | Plan and trial state | closed |
| Commercial 2 | Gateway boundary | closed |
| Commercial 3 | Enforce the three plans | closed |
| Commercial 4 | Operator surface | closed |
| Commercial 5 | Record and freeze | planned |

Follow-up phases 1–10 are closed. Full release history: [`docs/releases/`](docs/releases/)
Release timeline: [`docs/releases/timeline.md`](docs/releases/timeline.md)

---

## What is not claimed

- **No hosted service.** Buro Assistant is self-hosted by the operator.
  There is no SaaS offering, no shared infrastructure, and no uptime guarantee
  from Azimi Innovation Lab.
- **No commercial sale.** This repository is not a product offered for purchase
  or subscription.
- **No factory-ready certification.** Designed-for capabilities are marked in the
  table above. They are not verified unless a passing test exists in this repository.
- **No throughput guarantee.** Benchmark figures are from an in-memory SQLite
  development machine. Production PostgreSQL performance is the operator's
  responsibility. See [`docs/BENCHMARK.md`](docs/BENCHMARK.md).
- **No EU data-residency guarantee.** The database region is operator-controlled.
  See [`docs/PRIVACY_DATA_MAP.md`](docs/PRIVACY_DATA_MAP.md).
- **No MFA.** Multi-factor authentication is outside the scope of this codebase;
  it is the operator's hosting layer's responsibility.

---

## External links

- [FastAPI](https://fastapi.tiangolo.com/) — ASGI web framework used for the API layer
- [Alembic](https://alembic.sqlalchemy.org/) — database migration tool
- [SQLAlchemy](https://www.sqlalchemy.org/) — ORM and query builder
- [Argon2-cffi](https://argon2-cffi.readthedocs.io/) — password hashing
- [OpenAI API](https://platform.openai.com/docs) — model provider (requires operator key)
- [slowapi](https://github.com/laurentS/slowapi) — rate limiting for FastAPI

No private links, invitation-only URLs, or internal tooling URLs appear in this section.

---

## Author

Buro Assistant is an office mail desk built by Amin Azimi, AI Architect,
Azimi Innovation Lab. It receives tenant mail, classifies it, drafts a short
receipt, and places the work on the right queue so an operator can see what
arrived, what was decided, and what was sent.

Threat model: [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md)
Install guide: [`docs/INSTALL.md`](docs/INSTALL.md)
Runbook: [`docs/RUNBOOK.md`](docs/RUNBOOK.md)
Comparison: [`docs/COMPARISON.md`](docs/COMPARISON.md)
Test house: [`docs/TEST_HOUSE.md`](docs/TEST_HOUSE.md)
Install video script: [`docs/INSTALL_VIDEO_SCRIPT.md`](docs/INSTALL_VIDEO_SCRIPT.md)
