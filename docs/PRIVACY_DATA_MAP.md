# Buro Assistant — Privacy Data Map

**Product:** Buro Assistant  
**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab  
**Phase:** Follow-up Phase 4 — German-market privacy pack  
**Date:** 2026-10-06  

This document is the internal data map required under Art. 30 GDPR (records of
processing activities).  It lists every table and column that holds or may hold
personal data, the purpose of processing, the legal basis, and the default
retention period.

---

## Region choice

**The database region is operator-controlled.**  The operator sets the
`DATABASE_URL` environment variable to point to any database they host.
Buro Assistant does not operate a hosted service and makes no claim about
data residency.  If the operator must keep data in the EU, the operator
must host the database in an EU region.  The software does not enforce or
verify region automatically.

See `docs/INSTALL.md` for the full list of environment variables.

---

## Tables containing personal data

### `users`

| Column | Personal data | Purpose | Legal basis | Retention |
|--------|--------------|---------|-------------|-----------|
| `name` | Full name | Identify the operator account | Contract | Until account deleted by operator |
| `email` | Email address | Login credential, audit actor | Contract | Until account deleted |
| `password_hash` | Argon2id hash (not plaintext) | Authentication | Contract | Until account deleted |

No raw password is stored.  Session tokens are stored as SHA-256 hashes only
(`user_sessions.token_hash`).

### `messages`

| Column | Personal data | Purpose | Legal basis | Retention |
|--------|--------------|---------|-------------|-----------|
| `subject_normalized` | Possibly — subject line of incoming email | Message classification | Legitimate interest (inbox management) | `tenants.retention_days` (default 180 days) |
| `raw_json` | Possibly — envelope headers including sender/recipient addresses | Idempotency, audit trail | Legitimate interest | `tenants.retention_days` |
| `message_id_header` | RFC 5322 Message-Id (may contain domain) | Duplicate detection | Legitimate interest | Same |
| `legal_hold` | No | Blocks deletion when True | Legal obligation | Until hold lifted |

Body text of incoming messages is **not stored** in the `messages` table.
It is passed through the pipeline in memory, redacted before any model call,
and discarded.

### `drafts`

| Column | Personal data | Purpose | Legal basis | Retention |
|--------|--------------|---------|-------------|-----------|
| `subject` | Possibly — reply subject | Draft management | Contract | `tenants.retention_days` |
| `body` | Possibly — reply body (generated from template) | Draft management | Contract | Same |

Draft bodies are generated from approved templates.  The template system
prohibits fabricated facts (see `app/policy/templates.py`).

### `approval_queue`

| Column | Personal data | Purpose | Legal basis | Retention |
|--------|--------------|---------|-------------|-----------|
| `subject` | Possibly | Human review queue | Contract | `tenants.retention_days` |
| `body` | Possibly (truncated to 500 chars) | Human review queue | Contract | Same |
| `resolved_by` | Operator email | Accountability | Contract | Same |

### `audit_log`

| Column | Personal data | Purpose | Legal basis | Retention |
|--------|--------------|---------|-------------|-----------|
| `actor` | Operator email or "system" | Accountability, non-repudiation | Legal obligation | `tenants.retention_days` |
| `detail` | Possibly (operator-supplied free text) | Incident record | Legal obligation | Same |

### `usage_events`

| Column | Personal data | Purpose | Legal basis | Retention |
|--------|--------------|---------|-------------|-----------|
| `actor` | Operator email or "system" | Billing accountability | Contract | `tenants.retention_days` |

### `user_sessions`

| Column | Personal data | Purpose | Legal basis | Retention |
|--------|--------------|---------|-------------|-----------|
| `token_hash` | No (hash only) | Session authentication | Contract | 24 hours TTL |

### `api_keys`

| Column | Personal data | Purpose | Legal basis | Retention |
|--------|--------------|---------|-------------|-----------|
| `name` | Operator-supplied label | Key identification | Contract | Until revoked |
| `key_hash` | No (SHA-256 hash) | Key authentication | Contract | Until revoked |

### `webhook_subscriptions`

| Column | Personal data | Purpose | Legal basis | Retention |
|--------|--------------|---------|-------------|-----------|
| `url` | Possibly (endpoint URL may contain operator domain) | Event delivery | Contract | Until deleted |
| `secret_hash` | No (hash only) | Payload signing | Contract | Until deleted |

---

## Tables not containing personal data

The following tables hold operational data only and do not store personal data
as defined by Art. 4(1) GDPR:

- `tenants` — slug and name are administrative identifiers chosen by the operator
- `work_queue` — job payloads (opaque, provider_message_id only)
- `dead_letter` — failure metadata, no message body
- `quotas` — token/cost counters per tenant per day
- `decisions` — classification metadata (department, urgency, rule_hit); no body text
- `delivery_log` — webhook delivery status, payload preview (no PII by design)

---

## Redaction before model calls

All incoming message subjects and bodies are passed through `app/agents/redact.py`
before being sent to any model.  The following patterns are replaced with
placeholders before the model sees them:

| Pattern | Placeholder |
|---------|-------------|
| Email addresses | `[EMAIL]` |
| Phone numbers | `[PHONE]` |
| Credit card sequences | `[CARD]` |
| SSN/national ID sequences | `[SSN]` |
| IBAN numbers | `[IBAN]` |

Attachment bytes are **never** sent to a model.

---

## Right to erasure (Art. 17 GDPR)

`DELETE /desk/privacy/data` calls `app/domain/retention.delete_tenant_data()`.

What is deleted:
- Messages (unless `legal_hold=True`)
- Drafts
- Approval queue entries
- Audit log entries
- Usage events
- Work queue items
- Dead-letter items

What is retained:
- User accounts and tenant record (operator must delete separately)
- Messages with `legal_hold=True` (legal hold overrides erasure)

---

## Data portability (Art. 20 GDPR)

`GET /desk/privacy/export` calls `app/domain/export.export_tenant()` and
returns a JSON document containing all tenant data (excluding `raw_json`
by default).  The operator can download this and provide it to a data subject.

---

## Legal hold

Setting `messages.legal_hold = True` prevents a row from being deleted by
both the scheduled retention purge (`apply_retention`) and the right-to-erasure
endpoint (`delete_tenant_data`).  This protects evidence that must be preserved
for legal proceedings.  Only an operator with direct database access can set or
clear this flag (no desk endpoint modifies it — by design).

---

## Logs and secrets

- `IMAP_PASSWORD` and `IMAP_USER` are never logged.  A `_SecretFilter`
  is installed on the `app.workers.intake_loop` logger to scrub these
  values from any accidental future log record.
- Session tokens are never logged; only their SHA-256 hashes are stored.
- Webhook signing secrets are shown once at creation and never stored in
  plaintext; only their SHA-256 hash is in the database.
- No secret, key, or password appears in any sample file, seed script output
  visible in logs, or version-controlled file.

---

*End of data map.  For questions contact: Amin Azimi, AI Architect, Azimi Innovation Lab.*
