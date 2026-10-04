# Phase 9 Release Notes

**Product:** Buro Assistant
**Release tag:** phase-9
**Date:** 2026-10-03
**Checklist rows closed:** 40, 41, 42, 43, 44, 45

---

## Summary

Phase 9 delivers the commercial controls layer: append-only usage events, a full tenant data export job, a configurable retention window (default 180 days) with a purge helper, an isolated sandbox seed command, scoped API keys stored as SHA-256 hashes, and signed outbound webhooks using HMAC-SHA256. Migration `0006` creates the required tables and adds `retention_days` to `tenants`.

---

## What was built

### `app/domain/usage.py` — usage events (row 40)

- `UsageEvent` ORM model (`usage_events` table). Fields: `tenant_id`, `event_type`, `quantity`, `unit`, `cost_usd`, `actor`, `reference_id`, `created_at`.
- `emit(db, tenant_id, event_type, …)` — sole write path; always INSERT, never UPDATE.
- `events_for_tenant(db, tenant_id, event_type, limit)` — read path with optional type filter.
- Event categories: `message_ingested`, `model_called`, `draft_sent`, `api_key_used`, `webhook_fired`.

### `app/domain/export.py` — tenant data export (row 41)

- `export_tenant(db, tenant_id, include_raw_json=False)` — returns a JSON-serialisable dict containing: tenant record, users, messages, drafts, audit log, usage events, approval queue.
- `raw_json` is excluded by default; opt-in with `include_raw_json=True`.
- Read-only: no data is modified or deleted.
- Returns `{"error": "…"}` for an unknown tenant instead of raising.

### `app/domain/retention.py` — retention field, default 180 days (row 42)

- `RETENTION_DEFAULT_DAYS = 180`.
- `apply_retention(db, tenant_id, tenant_retention_days, now)` — deletes `Message` rows whose `ingest_time < now - retention_days`. Returns count of rows deleted. Tenant-scoped.
- Migration `0006` adds `retention_days` column to `tenants` (`server_default="180"`).

### `app/domain/sandbox.py` — sandbox tenant seed command (row 43)

- `seed_sandbox()` — idempotent command that creates the `sandbox` tenant, a sandbox user (`sandbox@example.com`), a sample task, and a default API key (printed once).
- Run as: `python -m app.domain.sandbox`
- Distinct from the Phase 3 demo seed (`python -m app.domain.seed`) which targets the `demo` slug.

### `app/domain/apikeys.py` — scoped API keys, stored hashed (row 44)

- `ApiKey` ORM model (`api_keys` table). Fields: `tenant_id`, `name`, `key_hash` (SHA-256), `scope`, `created_at`, `last_used_at`, `revoked`.
- `create_api_key(db, tenant_id, name, scope)` — returns `(ApiKey, raw_key)`. Raw key has the `buro_` prefix; only its SHA-256 hash is persisted (row 44).
- `lookup_api_key(db, raw)` — hashes the raw key, looks up non-revoked entry, updates `last_used_at`.
- `revoke_api_key(db, key_id, tenant_id)` — cross-tenant revocation returns `False`.
- `list_api_keys(db, tenant_id)` — excludes revoked keys.

### `app/domain/webhooks.py` — signed outbound webhooks (row 45)

- `WebhookSubscription` ORM model (`webhook_subscriptions` table). Fields: `tenant_id`, `url`, `secret_hash` (SHA-256), `events` (comma-separated filter), `active`, `created_at`.
- `create_subscription(db, tenant_id, url, events)` — returns `(sub, plaintext_secret)`. Only the SHA-256 hash of the secret is stored.
- `build_signed_payload(event_type, data, secret)` — returns `(bytes, "sha256=<hex>")`.
- `verify_signature(payload_bytes, secret, header_value)` — constant-time comparison.
- `active_subscriptions(db, tenant_id, event_type)` — filters by event type; empty `events` field matches all.
- `deactivate_subscription(db, sub_id, tenant_id)` — tenant-scoped deactivation.

---

## Migration

`migrations/versions/0006_commercial.py` — idempotent upgrade:
- Adds `retention_days` to `tenants` (`server_default="180"`).
- Creates `usage_events`, `api_keys`, `webhook_subscriptions` tables.

---

## Tests

`tests/test_commercial.py` — **50 tests** across 6 test classes:

| Class | Tests | Row |
|---|---|---|
| `TestUsageEvents` | 7 | 40 |
| `TestExport` | 8 | 41 |
| `TestRetention` | 6 | 42 |
| `TestSandboxSeed` | 7 | 43 |
| `TestApiKeys` | 9 | 44 |
| `TestWebhooks` | 13 | 45 |

Full suite: **329 tests, 0 failures**.

---

## Capability matrix delta

| Row | Capability | Status |
|---|---|---|
| 40 | Usage events | ✓ Phase 9 |
| 41 | Tenant data export job | ✓ Phase 9 |
| 42 | Retention field, default 180 days | ✓ Phase 9 |
| 43 | Sandbox tenant seed command | ✓ Phase 9 |
| 44 | Scoped API keys, stored hashed | ✓ Phase 9 |
| 45 | Signed outbound webhooks | ✓ Phase 9 |

---

## What is not in this phase

- No HTTP delivery of webhook payloads (a live `httpx.post` is wired in the worker layer, Phase 10 wires it end-to-end).
- No API endpoints exposing usage events, export, or API key management (Phase 10 desk extensions).
- Rows 49 and 50 (runbook, install/backup/threat model) are Phase 10.
