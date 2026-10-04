# Follow-up Phase 6 — Cinematic README and release surface

**Product:** Buro Assistant  
**Contract:** Follow-up contract (development restart 2026-10-04)  
**Phase:** 6 — Cinematic README and release surface  
**Date:** 2026-10-08  
**Branch:** main  

---

## What this phase does

Rewrites `README.md` and adds `docs/diagrams/` to give the project a release
surface that a buyer can scan in one minute and that satisfies the honesty rule
(designed-for is not claimed as verified).

```
README.md
  – Author line: Amin Azimi, AI Architect, Azimi Innovation Lab
  – Current version and license reference at the top
  – Contents table of contents with anchor links (all internal)
  – "What it does" section with key safety invariants stated
  – Architecture diagram embedded in a code fence
  – Link to docs/diagrams/architecture.txt
  – Agents section with internal links to schemas/
  – File map table: one responsibility per package
  – Designed versus verified table
      · Every verified capability cites the test file
      · Three unverified capabilities marked "designed for — not yet
        externally measured" (live mailbox poll, horizontal scale,
        quota at scale)
  – Quick start: git clone, cp .env.example, pip install, ./run.sh
  – Configuration table with database-region disclaimer
  – Tests table: 454 tests, human-readable coverage by file
  – Roadmap table: follow-up phases 6–10 with status
  – "What is not claimed" section (no hosted service, no commercial
    sale, no factory-ready cert, no throughput guarantee, no EU
    residency guarantee, no MFA)
  – External links section (FastAPI, Alembic, SQLAlchemy, Argon2-cffi,
    OpenAI API, slowapi) — no private links
  – Author section with links to THREAT_MODEL, INSTALL, RUNBOOK

docs/diagrams/architecture.txt
  – ASCII-art architecture diagram covering all layers:
    external mail server → ingest → pipeline (Amin → Amilos / Leila)
    → policy → workers → operator desk → data layer → security boundaries
```

---

## Test command

```bash
python3 -m pytest tests/ -q
```

**Result:** 454 passed, 0 failed, 0 errors.

No new tests were written. Phase 6 is documentation only. The existing 454 tests
verify that the codebase the README describes is still correct.

---

## Requirements satisfied

| Requirement | Satisfied | Notes |
|---|---|---|
| Author Amin Azimi, Azimi Innovation Lab | yes | Top of README |
| Internal links only in body | yes | All `href` targets are file paths or anchor fragments; no private URLs |
| External-links section | yes | Six public project links; no private or invitation-only URLs |
| Roadmap | yes | Table of follow-up phases 6–10 with status |
| Designed-versus-verified table | yes | Every verified capability cites a test file; three unverified capabilities explicitly marked |
| Architecture diagram in text | yes | Embedded code fence in README + `docs/diagrams/architecture.txt` |
| Quick start | yes | git clone → cp .env.example → pip install → ./run.sh |
| What is not claimed | yes | Dedicated section: no hosted service, no commercial sale, no cert, no throughput guarantee, no EU guarantee, no MFA |
| docs/diagrams created | yes | `docs/diagrams/architecture.txt` |
| No private links | yes | Reviewed: no internal tooling, no invitation-only URLs |
| App stays runnable | yes | 454 tests pass; no code changed |

---

## Files changed

| File | Change |
|---|---|
| `README.md` | Full rewrite — see section above for all additions |
| `docs/diagrams/architecture.txt` | **New.** Full ASCII architecture diagram covering all layers and security boundaries |
| `docs/releases/follow-up-phase-6.md` | **New.** This file |

---

## Test failures

None. All 454 tests pass.
