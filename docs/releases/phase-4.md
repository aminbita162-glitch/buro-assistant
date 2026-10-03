# Phase 4 Release Notes

**Product:** Buro Assistant
**Release tag:** phase-4
**Date:** 2026-10-03
**Checklist rows closed:** 3, 4, 5, 6, 9, 12

---

## Summary

Phase 4 delivers the ingest pipeline: every mail message that arrives is normalised, deduplicated, stored immutably with its raw payload, and quarantined if it carries a disallowed attachment type. The pipeline is provider-neutral; the IMAP adapter is the first concrete provider and reads credentials exclusively from environment variables.

---

## What was built

### `app/ingest/models.py` — Message table (rows 3, 4, 12)
- `Message` ORM model persisted to the `messages` table.
- `UNIQUE(tenant_id, provider_message_id)` constraint enforces idempotency: re-delivering the same provider message for the same tenant is a no-op (row 3).
- `raw_json` column stores the full provider payload as immutable JSON; it is written once and never updated (row 4).
- `attachment_state` column carries `allowed`, `quarantined`, or `none`; controlled by an allowlist of safe MIME types (row 12).
- `state` column tracks the lifecycle: `received → processing → classified → replied → failed`.
- Alembic migration `0002_ingest.py` creates the table; `app/main.py` never calls `create_all`.

### `app/ingest/normalize.py` — NormalizedMessage (row 6)
- `NormalizedMessage` is a frozen `dataclass` that holds provider-neutral fields: `provider`, `provider_message_id`, `subject_normalized`, `message_id_header`, `body_text`, `attachment_names`, `attachment_types`, `raw_payload`.
- `subject_normalized` is lowercased and stripped so duplicate detection is case- and whitespace-insensitive (row 9).

### `app/ingest/providers/base.py` — Abstract MailProvider (rows 5, 6)
- `MailProvider` ABC defines `fetch_messages(limit)` returning `list[NormalizedMessage]` and `mark_seen(provider_message_id)`.
- All concrete adapters must satisfy this contract.

### `app/ingest/providers/imap_provider.py` — IMAP adapter (row 5)
- Connects to any IMAP server using `IMAP_HOST`, `IMAP_PORT`, `IMAP_USER`, `IMAP_PASSWORD` from the environment only; no credential arguments accepted at construction time.
- Fetches unseen messages, parses headers, decodes `text/plain` parts, extracts attachment filenames and MIME types, and emits `NormalizedMessage` objects.
- `mark_seen` sets the `\Seen` flag on the server.

### `app/ingest/providers/fake_provider.py` — In-memory fake (test support)
- `FakeProvider` pre-loads a list of `NormalizedMessage` objects and pops them on `fetch_messages`; `mark_seen` is a no-op.
- Used exclusively in tests; no production code path references it.

### `app/ingest/ingest.py` — `ingest_message()` orchestrator (rows 3, 4, 6, 9, 12)
- Accepts a `Session`, `tenant_id`, and a `NormalizedMessage`.
- Deduplicates by `(tenant_id, provider_message_id)` (row 3).
- Deduplicates by `(tenant_id, message_id_header)` when the header is present, and by `(tenant_id, subject_normalized)` when the subject matches an existing message within the same tenant (row 9).
- Stores `raw_json` once; subsequent calls for the same key return the existing record (row 4).
- Evaluates attachment MIME types against `ATTACHMENT_ALLOWLIST`; sets `attachment_state = quarantined` for any disallowed type (row 12).
- Returns the `Message` ORM object (new or existing).

---

## Migration

`migrations/versions/0002_ingest.py` — creates the `messages` table with all columns and the `uq_tenant_provider_msg` unique constraint. Applied automatically by `_run_migrations()` on startup.

---

## Tests

`tests/test_ingest.py` — 26 tests across five test classes:

| Class | Tests |
|---|---|
| `TestMessageModel` | Column presence, unique constraint, raw_json immutability |
| `TestNormalizedMessage` | Dataclass fields, normalization, frozen enforcement |
| `TestFakeProvider` | fetch/mark_seen contract |
| `TestIngestMessage` | New message stored, idempotency by provider id, dedup by Message-Id header, dedup by subject, attachment allowlist / quarantine, state and attachment_state defaults |
| `TestIMAPProviderConfig` | Env-only credential enforcement (no live server required) |

Full suite: **77 tests, 0 failures**.

---

## Capability matrix delta

| Row | Capability | Status |
|---|---|---|
| 3 | Idempotency key of provider message id plus tenant | ✓ Phase 4 |
| 4 | Immutable stored raw message | ✓ Phase 4 |
| 5 | IMAP adapter, credentials only from environment | ✓ Phase 4 |
| 6 | Provider-neutral normalized message | ✓ Phase 4 |
| 9 | Duplicate detection by Message-Id and normalized subject | ✓ Phase 4 |
| 12 | Attachment allowlist and quarantine state | ✓ Phase 4 |

---

## What is not in this phase

- No live IMAP polling loop (Phase 8 worker).
- No agent decisions on ingested messages (Phase 5).
- No API endpoints exposing the messages table (Phase 7).
