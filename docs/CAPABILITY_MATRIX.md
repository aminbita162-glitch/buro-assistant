# Capability Matrix

**Product:** Buro Assistant
**Release target:** 1.0.0
**Development restart:** 2026-10-03

This matrix lists every capability defined in the release checklist. The **Designed for** column marks capabilities that are part of the architecture. The **Verified** column marks capabilities confirmed by a passing test in this repository. Numbers in parentheses refer to checklist rows in DIRECTIVE.txt.

| # | Capability | Designed for | Verified |
|---|---|---|---|
| 1 | Tenant on every business table | Phase 3 | — |
| 2 | Tests that reject cross-tenant reads | Phase 3 | — |
| 3 | Idempotency key of provider message id plus tenant | Phase 4 | — |
| 4 | Immutable stored raw message | Phase 4 | — |
| 5 | IMAP adapter, credentials only from environment | Phase 4 | — |
| 6 | Provider-neutral normalized message | Phase 4 | — |
| 7 | Tenant rule pack for domain, subject, and department | Phase 5 | — |
| 8 | Confidence threshold; below threshold goes to Supervisor | Phase 5 | — |
| 9 | Duplicate detection by Message-Id and normalized subject | Phase 4 | — |
| 10 | Language detection by library first | Phase 5 | — |
| 11 | Urgency lexicon with tenant overrides | Phase 5 | — |
| 12 | Attachment allowlist and quarantine state | Phase 4 | — |
| 13 | Redaction before any model call | Phase 5 | — |
| 14 | Prompt and schema version stored on each decision | Phase 5 | — |
| 15 | Schema validation of agent output | Phase 5 | — |
| 16 | Template registry with variables and forbidden phrases | Phase 6 | — |
| 17 | Auto-reply off unless tenant turns it on | Phase 6 | — |
| 18 | Receipt template stating message was received and will be reviewed | Phase 6 | — |
| 19 | Business-hours calendar per tenant | Phase 6 | — |
| 20 | SLA clock from ingest time | Phase 6 | — |
| 21 | Human approval queue | Phase 6 | — |
| 22 | Append-only audit log | Phase 6 | — |
| 23 | Dashboard counts: received, classified, drafted, sent, held, failed | Phase 7 | — |
| 24 | Department queues | Phase 7 | — |
| 25 | Cost and token fields on model calls | Phase 8 | — |
| 26 | Per-tenant daily quota | Phase 8 | — |
| 27 | Backpressure when queue depth exceeds tenant cap | Phase 8 | — |
| 28 | Dead-letter queue and replay | Phase 8 | — |
| 29 | Priority lanes | Phase 8 | — |
| 30 | Traces around ingest, decide, draft, and send | Phase 8 | — |
| 31 | Live and ready health checks; ready checks the database | Phase 2 | — |
| 32 | Alembic migrations; no schema change on import | Phase 3 | — |
| 33 | Argon2id password hashes with one-time legacy SHA-256 verification | Phase 2 | — |
| 34 | Session table, hashed token, expiry, rotation on login | Phase 2 | — |
| 35 | Rate limits on signup, login, and model routes | Phase 2 | — |
| 36 | Security headers and a content security policy | Phase 2 | — |
| 37 | CORS allowlist from the environment | Phase 2 | — |
| 38 | Message and task fields rendered as text, not HTML | Phase 2 | — |
| 39 | Capability matrix with Designed for and Verified columns | ✓ Phase 1 | ✓ Phase 1 |
| 40 | Usage events | Phase 9 | — |
| 41 | Tenant data export job | Phase 9 | — |
| 42 | Retention field, default 180 days | Phase 9 | — |
| 43 | Sandbox tenant seed command | Phase 9 | — |
| 44 | Scoped API keys, stored hashed | Phase 9 | — |
| 45 | Signed outbound webhooks | Phase 9 | — |
| 46 | Fifty synthetic golden messages with expected decisions | Phase 5 | — |
| 47 | Contract tests for the three schemas | Phase 5 | — |
| 48 | Shadow mode that stores a draft and does not send | Phase 6 | — |
| 49 | Runbook for incident, quota breach, and bad template | Phase 10 | — |
| 50 | Install, backup, restore, threat model, and release notes | Phase 10 | — |

---

*This matrix is updated at the end of each phase. A Verified entry is only written when a test in this repository passes for that row.*
