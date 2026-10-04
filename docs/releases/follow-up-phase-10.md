# Follow-up Phase 10 — External proof and freeze

**Product:** Buro Assistant  
**Contract:** Follow-up contract (development restart 2026-10-04)  
**Phase:** 10 — External proof and freeze  
**Date:** 2026-10-11  
**Branch:** main  

---

## What this phase does

Writes the external test script that a second account can run independently,
records the result of that run, and freezes the roadmap.

No new feature code is added this phase.

---

## Files changed

| File | Change |
|---|---|
| `docs/EXTERNAL_TEST.sh` | **New.** Shell script — install, seed, send a fake message, check the desk, fail a cross-tenant read |
| `docs/TEST_HOUSE.md` | Added external test entry (not run) and run #11 (Phase 10) |
| `docs/releases/follow-up-phase-10.md` | **New.** This file — Phase 10 report |

---

## External test script

The script is at [`docs/EXTERNAL_TEST.sh`](../EXTERNAL_TEST.sh).

### What the script checks

| Step | Action | Expected result |
|---|---|---|
| 1 | `pip install -r requirements.txt` | exits 0 |
| 2 | `python3 -m pytest tests/ -q` | all tests pass |
| 3 | `python3 -m app.domain.seed` | demo tenant and user created |
| 4 | `uvicorn app.main:app` | server starts on port 8000 |
| 5 | `GET /health` | HTTP 200 |
| 6 | `POST /auth/signup` (tenant-A) | operator created |
| 7 | `POST /auth/login` (tenant-A) | session token returned |
| 8 | `POST /ingest` with fake message | `result=new` |
| 9 | `GET /desk/queue` as tenant-A | `count ≥ 1` |
| 10 | `POST /auth/signup` + login (tenant-B) | second operator created |
| 11 | `GET /desk/queue` as tenant-B | `count = 0` (cross-tenant read blocked) |

### Prerequisites

- Python 3.11 or later
- A PostgreSQL instance, or `DATABASE_URL=sqlite:///buro_test.db` for a local run
- `.env` copied from `.env.example` with `DATABASE_URL` and `OPENAI_API_KEY` set

### How to run

```bash
cp .env.example .env          # fill in DATABASE_URL and OPENAI_API_KEY
bash docs/EXTERNAL_TEST.sh
```

---

## External test result

**not run**

This script has not been executed by a second account.
When a second account runs it, the runner records date, actor, and result
in `docs/TEST_HOUSE.md` under "External test runs".
The claim "a second account has verified this" is not made until that entry
is written.

---

## Roadmap freeze

The roadmap is now frozen at this point.
What is done and what is not done is recorded accurately below.

### Completed (1.1.0)

| Item | Status |
|---|---|
| End-to-end pipeline (ingest → Amin → Amilos/Leila → shadow/approval) | Verified — tests pass |
| IMAP worker loop with fake-adapter fallback | Verified — tests pass |
| Signed webhook delivery with retries and delivery log | Verified — tests pass |
| Department stored as field; desk filtered by department | Verified — tests pass |
| GDPR export, erasure, retention enforcement, legal hold | Verified — tests pass |
| Rule-before-model triage; zero tokens on rule hit | Verified — tests pass |
| Attachment bytes excluded from all model prompts | Verified — tests pass |
| Prompt cache for identical inputs | Verified — tests pass |
| Per-tenant daily token quota dashboard | Verified — tests pass |
| Cinematic README; architecture diagram; designed-vs-verified table | Verified — tests pass |
| All 15 Section B buyer options | Designed for — tests pass; not externally verified |
| All 15 Section C finish items | Verified — tests pass |
| External test script | Written — not yet run by a second account |

### Not done and not claimed

| Item | Note |
|---|---|
| Commercial subscription or billing | Out of scope |
| Hosted service | Out of scope |
| External test run by a second account | Script written; run not recorded |
| Factory or commercial installation recommendation | Not made until external test is recorded |
| Throughput or latency figures | Not measured |
| Cost comparison against competing products | Not made |

---

## Test command

```bash
python3 -m pytest tests/ -q
```

**Result:** 600 passed, 0 failed, 0 errors.

Baseline unchanged from Phase 9.  No new tests were added this phase.

---

## Test failures

None.  All 600 tests pass.

---

## Tag

No release tag has been created.  The operator creates the tag when ready.

```bash
git tag v1.1.0
git push origin v1.1.0
```

---

*This is the final phase of the follow-up contract.  No Phase 11 will be started.*  
*For questions contact: Amin Azimi, AI Architect, Azimi Innovation Lab.*
