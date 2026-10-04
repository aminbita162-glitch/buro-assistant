# Buro Assistant — Threat Model

**Product:** Buro Assistant  
**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab  
**Phase:** Follow-up Phase 4 — German-market privacy pack  
**Date:** 2026-10-06  
**Methodology:** STRIDE (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege)  

---

## Scope

This document covers the Buro Assistant back-end API and the operator desk
front-end (`index.html`).  It does not cover the operator's mail server,
cloud hosting environment, or network perimeter — those are the operator's
responsibility.

**Not in scope:** a hosted multi-tenant service.  Buro Assistant is designed
for operator-controlled self-hosting.  Commercial sale and a hosted service are
not done (see `docs/PRIVACY_DATA_MAP.md` and `README.md`).

---

## Trust boundaries

```
[Sender's mail server]
        │  IMAP / SMTP
        ▼
[IMAPProvider / FakeProvider]  ← credentials from env only
        │
        ▼
[Ingest layer]  →  [Pipeline]  →  [DB (operator-hosted)]
                        │
                        ▼
                  [Model API]  ← OPENAI_API_KEY from env only
                        │
                        ▼
                  [Operator Desk (index.html)]  ← Bearer token from sessionStorage
                        │
                        ▼
                  [Webhook targets]  ← signed payload, HMAC-SHA256
```

---

## STRIDE analysis

### S — Spoofing

| Asset | Threat | Mitigation |
|-------|--------|-----------|
| Operator login | Credential stuffing | Argon2id password hashing; session rotation on every login; 10/min rate limit on `/auth/login` |
| API key auth | Key theft | Keys stored as SHA-256 hashes only; plaintext shown once at creation |
| Webhook payload | Forged delivery claim | Payloads signed with HMAC-SHA256; receiver can verify `X-Buro-Signature` |
| IMAP login | Credential exposure | Password read from `IMAP_PASSWORD` env var only; never logged, never stored |

### T — Tampering

| Asset | Threat | Mitigation |
|-------|--------|-----------|
| Audit log | Log manipulation | Append-only table; no UPDATE/DELETE path in code |
| Draft content | Injection via incoming email | Body and subject redacted before model call; template system enforces structure |
| Message raw_json | Stored XSS | `raw_json` is never rendered in the desk UI; desk uses `textContent` only |

### R — Repudiation

| Asset | Threat | Mitigation |
|-------|--------|-----------|
| Approval decisions | "I didn't approve that" | `resolved_by` field records the operator email; audit log row written |
| Ingest events | Dispute about receipt | `ingest_time` recorded at ingest; `provider_message_id` is idempotency key |

### I — Information Disclosure

| Asset | Threat | Mitigation |
|-------|--------|-----------|
| IMAP password | Leaked via logs | `_SecretFilter` on intake-loop logger; `IMAP_PASSWORD` never interpolated into log format strings |
| Session tokens | Token theft from logs | Tokens never logged; only SHA-256 hashes stored |
| Other-tenant data | Cross-tenant read | Every query filters by `user.tenant_id`; desk endpoints enforce tenant scope |
| Model API key | Leaked in response | `OPENAI_API_KEY` read at startup only; never included in any response |
| PII in model prompts | Sent to third-party model | Redaction applied before every model call (`app/agents/redact.py`) |
| Attachment bytes | Sent to model | Attachment bytes are never included in any model prompt |

**Known gap:** The model API (OpenAI) is a third-party service.  Even after
redaction, the subject and body clip are transmitted to OpenAI's servers.
Operators with data-residency requirements must evaluate whether this is
acceptable or switch to an on-premises model.  This is documented, not hidden.

### D — Denial of Service

| Asset | Threat | Mitigation |
|-------|--------|-----------|
| `/assistant`, `/auth/login` | Brute-force / flood | `slowapi` rate limiter on sensitive endpoints |
| Work queue | Queue flooding | Per-tenant `queue_depth_cap` (default 500); `BackpressureError` rejects over-limit enqueues |
| Token budget | Runaway model calls | Per-tenant daily token quota (`daily_token_quota`); `QuotaExceeded` blocks model calls |
| Dead letter | DLQ growth | DLQ is observable; operator runbook covers replay |

### E — Elevation of Privilege

| Asset | Threat | Mitigation |
|-------|--------|-----------|
| Admin functions | Unprivileged access | All desk endpoints require a valid Bearer token scoped to the authenticated tenant |
| Cross-tenant admin | Tenant A sees tenant B | Every query filters by `user.tenant_id`; no global admin endpoint exists |
| Legal hold bypass | Operator circumvents hold | `legal_hold` can only be set/cleared via direct DB access — no API endpoint modifies it |

---

## Residual risks

| Risk | Severity | Accepted / Mitigated |
|------|----------|---------------------|
| OpenAI receives redacted text | Medium | Accepted — operator must evaluate; documented in data map |
| No RBAC (admin / operator / auditor roles) | Medium | Accepted — planned for Phase 7 (Section B item 13) |
| No MFA on operator login | Low–Medium | Accepted — outside scope of this codebase; operator's hosting layer may add MFA |
| SQLite in development has no encryption at rest | Low | Accepted — operators must use an encrypted database in production |
| `raw_json` contains envelope headers | Low | Mitigated — never rendered in UI; not included in export by default |

---

## Out-of-scope threats

- Network-level attacks (DDoS, TLS downgrade) — mitigated by the operator's infrastructure
- Physical access to the database server — operator's responsibility
- Supply-chain attacks on Python dependencies — operator's responsibility (pin dependencies, run audits)
- Model prompt injection by sophisticated attackers — partial mitigation via redaction and templates; full mitigation requires model-side controls

---

## Changes in Phase 4

- `_SecretFilter` added to `app/workers/intake_loop` logger (I — information disclosure)
- `legal_hold` column added to `messages`; retention and erasure functions skip held rows (R — repudiation / legal obligation)
- `GET /desk/privacy/export` and `DELETE /desk/privacy/data` added to support GDPR Art. 17 and 20
- This threat model written

---

*For questions contact: Amin Azimi, AI Architect, Azimi Innovation Lab.*
