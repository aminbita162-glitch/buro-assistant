# Follow-up Phase 9 — Test house and release (1.1.0)

**Product:** Buro Assistant  
**Contract:** Follow-up contract (development restart 2026-10-04)  
**Phase:** 9 — Test house and release  
**Date:** 2026-10-10  
**Branch:** main  

---

## What this phase does

Records the Phase 9 test-house entry, corrects the license line in
`docs/COMPARISON.md`, and writes the 1.1.0 release notes.
No new feature code or tests are added this phase.

---

## Files changed

| File | Change |
|---|---|
| `docs/COMPARISON.md` | Corrected "MIT" license claim to proprietary all-rights-reserved license |
| `docs/TEST_HOUSE.md` | Added run #10 (Phase 9 run); updated last-updated header |
| `docs/releases/follow-up-phase-9.md` | **New.** This file — Phase 9 report and 1.1.0 release notes |

---

## License correction

The comparison table in `docs/COMPARISON.md` previously stated `✓ (MIT)` for
the "Open-source / inspectable" row. The actual repository license is
proprietary and all rights are reserved by Amin Azimi (see `LICENSE`).
The row now reads:

> ✓ (proprietary — all rights reserved, Amin Azimi)

No MIT grant exists. No other person receives a redistribution right unless
Amin Azimi grants it separately.

---

## Release notes — version 1.1.0

### Closed gaps (relative to 1.0.0)

| Gap recorded at 1.0.0 | Closed in phase | Notes |
|---|---|---|
| No live mailbox poll | Follow-up Phase 2 | IMAP worker loop with fake adapter fallback |
| No live webhook delivery | Follow-up Phase 2 | Signed delivery, retries, delivery log |
| Department filter was only a subject keyword | Follow-up Phase 3 | Department is now a stored field |
| README was not the release surface | Follow-up Phase 6 | README rewritten; cinematic opening, architecture diagram, roadmap |
| No German-market privacy pack | Follow-up Phase 4 | Data map, retention enforcement, export, delete, redaction, legal hold |
| No external test record | Follow-up Phase 9 | External test script planned for Phase 10; "not run" is written, not hidden |
| No release tag | Follow-up Phase 9 | Tag is not created unless the operator asks (see DIRECTIVE.txt) |

### New capabilities in 1.1.0

| Capability | Introduced | Status |
|---|---|---|
| End-to-end pipeline (ingest → Amin → Amilos/Leila → shadow/approval) | Phase 1 | Verified — tests pass |
| IMAP worker loop with fake-adapter fallback | Phase 2 | Verified — tests pass |
| Signed webhook delivery with retries and delivery log | Phase 2 | Verified — tests pass |
| Department stored as field; desk filtered by department | Phase 3 | Verified — tests pass |
| GDPR export, erasure, retention enforcement, legal hold | Phase 4 | Verified — tests pass |
| Rule-before-model triage; zero tokens on rule hit | Phase 5 | Verified — tests pass |
| Attachment bytes excluded from all model prompts | Phase 5 | Verified — tests pass |
| Prompt cache for identical inputs | Phase 5 | Verified — tests pass |
| Per-tenant daily token quota dashboard | Phase 5 | Verified — tests pass |
| Cinematic README; architecture diagram; designed-vs-verified table | Phase 6 | Verified — tests pass |
| All 15 Section B buyer options (assignment, notes, collision lock, snooze, VIP, after-hours, snippets, SLA, CSV export, bounce display, vacation responder, legal hold, role split, templates, digest) | Phase 7 | Designed for — tests pass; not externally verified |
| All 15 Section C finish items (README scan, diagram, comparison page, threat model link, test-house index, timeline, keyboard shortcuts, consistent names, install video script) | Phase 8 | Verified — tests pass |

### Remaining commercial work (not done in 1.1.0)

- Commercial subscription or billing is not implemented.
- Hosted service is not offered.
- External test by a second account has not been run (planned Phase 10).
- Factory or commercial installation is not recommended until Phase 10 external test is recorded.

### What is not claimed

- No throughput figure has been measured.
- No latency figure has been measured.
- No cost comparison has been made against competing products.
- "A second account can install this" is not claimed until Phase 10 records that run.

---

## Test command

```bash
python3 -m pytest tests/ -q
```

**Result:** 600 passed, 0 failed, 0 errors.

Baseline unchanged from Phase 8. No new tests were added this phase.

---

## Test failures

None. All 600 tests pass.

---

## Tag

No release tag has been created. The operator creates the tag when ready.
To tag from the command line:

```bash
git tag v1.1.0
git push origin v1.1.0
```

---

*For questions contact: Amin Azimi, AI Architect, Azimi Innovation Lab.*
