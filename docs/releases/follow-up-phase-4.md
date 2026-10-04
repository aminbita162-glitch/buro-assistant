# Follow-up Phase 4 — German-market privacy pack

**Product:** Buro Assistant  
**Contract:** Follow-up contract (development restart 2026-10-04)  
**Phase:** 4 – German-market privacy pack  
**Date:** 2026-10-06  
**Branch:** main  

---

## What this phase does

Implements the four required privacy-pack components and the two cross-cutting
requirements from the DIRECTIVE.txt stop condition:

```
Data map         docs/PRIVACY_DATA_MAP.md
                   – every table/column with personal data, purpose, legal basis,
                     retention period, redaction note, legal-hold note
                   – region choice documented as operator-controlled (not a hosted EU claim)

Retention        app/domain/retention.apply_retention()
enforcement        – legal_hold=True blocks deletion (Section B item 12)
                   – messages.legal_hold column added (migration 0009)

Export           GET /desk/privacy/export
(Art. 20)          – calls domain/export.export_tenant(); raw_json excluded by default
                   – tenant-scoped; returns messages, drafts, audit, usage, approval queue

Delete           DELETE /desk/privacy/data
(Art. 17)          – calls domain/retention.delete_tenant_data()
                   – legal-hold rows are skipped, counted in legal_hold_skipped
                   – tenant record and user accounts retained (operator action to delete)

Secrets/logs     app/workers/intake_loop._SecretFilter
                   – installed on intake-loop logger at module load time
                   – scrubs IMAP_PASSWORD and IMAP_USER values from any log record
                   – IMAP_PASSWORD was already not logged; filter is a defence-in-depth guard

Threat model     docs/THREAT_MODEL.md
                   – full STRIDE analysis (S, T, R, I, D, E)
                   – residual risks table with accepted / mitigated notes
                   – linked from docs/INSTALL.md
```

---

## Test command

```bash
python3 -m pytest tests/ -q
```

**Result:** 440 passed, 0 failed, 0 errors.

Previous baseline: 413 tests (Phase 3).  
New tests this phase: 27 (in `tests/test_privacy.py`).

---

## Requirements satisfied

| Requirement | Satisfied | Notes |
|---|---|---|
| Data map | yes | `docs/PRIVACY_DATA_MAP.md` — all tables, all personal columns, purpose, legal basis, retention |
| Retention enforcement | yes | `apply_retention` skips `legal_hold=True` rows; `messages.legal_hold` column added (migration 0009) |
| Export for one tenant | yes | `GET /desk/privacy/export` — tenant-scoped, JSON, excludes raw_json |
| Delete for one tenant | yes | `DELETE /desk/privacy/data` — tenant-scoped, legal-hold rows skipped |
| Redaction before model call stays | yes | Existing `app/agents/redact.py`; regression tests added |
| Logs must not print secrets or raw mailbox passwords | yes | `IMAP_PASSWORD` was never logged; `_SecretFilter` now guards against future regression |
| Region choice as operator-controlled, not hosted EU claim | yes | Documented in `docs/PRIVACY_DATA_MAP.md` and `docs/INSTALL.md`; no EU residency claim anywhere |
| Threat-model update | yes | `docs/THREAT_MODEL.md` — STRIDE analysis + Phase 4 change log |
| App stays runnable | yes | All existing routes and start-up sequence unchanged |

---

## Files changed

| File | Change |
|---|---|
| `migrations/versions/0009_legal_hold.py` | **New.** Adds `messages.legal_hold` Boolean column (default False). |
| `app/ingest/models.py` | `legal_hold = Column(Boolean, nullable=False, default=False)` added to `Message`. |
| `app/domain/retention.py` | `apply_retention` skips legal-hold rows; `delete_tenant_data()` added (right-to-erasure, counts deletions per table). |
| `app/workers/intake_loop.py` | `_SecretFilter` class and `logger.addFilter(_SecretFilter())` added; docstring updated. |
| `app/web/desk.py` | `GET /desk/privacy/export` and `DELETE /desk/privacy/data` endpoints added; `legal_hold` field added to `_ser_message`. |
| `docs/PRIVACY_DATA_MAP.md` | **New.** Art. 30 GDPR data map: all tables, personal columns, purpose, legal basis, retention, redaction, region note. |
| `docs/THREAT_MODEL.md` | **New.** Full STRIDE threat model with residual risks table. |
| `docs/INSTALL.md` | Region-choice note added to step 3; threat-model link added; commercial-sale out-of-scope note added. |
| `tests/test_privacy.py` | **New.** 27 tests covering export, erasure, legal hold, retention, secret filter, redaction. |
| `docs/releases/follow-up-phase-4.md` | **New.** This file. |

---

## Known limitations (not failures)

- `legal_hold` can only be set or cleared via direct database access; there is no
  API endpoint for this by design (only a DBA-level operator should modify it).
- `delete_tenant_data` retains user accounts and the tenant row.  Full account
  deletion is a separate operator action; it is not automated here.
- The model API (OpenAI) receives redacted text.  Redaction removes email,
  phone, IBAN, card, and SSN patterns but is not a guarantee against all PII.
  Operators with strict data-residency requirements should evaluate whether to
  use an on-premises model.  This risk is documented in `docs/THREAT_MODEL.md`.
- Region choice is operator-controlled.  The software does not enforce or verify
  EU residency.  This is documented, not hidden.

---

## Test failures

None. All 440 tests pass.
