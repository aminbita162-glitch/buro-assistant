# Follow-up Phase 8 — Differentiation pack (fifteen finish items)

**Product:** Buro Assistant  
**Contract:** Follow-up contract (development restart 2026-10-04)  
**Phase:** 8 — Differentiation pack (fifteen finish items, Section C)  
**Date:** 2026-10-09  
**Branch:** main  

---

## What this phase does

Implements all fifteen finish items from Section C of DIRECTIVE.txt.
Each item is a documentation, desk, or naming artifact with a passing test.

```
README.md                             – at-a-glance opening, file map, roadmap updated
docs/diagrams/architecture.txt        – last-updated bumped to Phase 8, migration count corrected
docs/COMPARISON.md                    – new: comparison against HelpScout, Front, Freshdesk
docs/releases/timeline.md             – new: full release timeline
docs/TEST_HOUSE.md                    – new: test-house index with all run records
docs/INSTALL_VIDEO_SCRIPT.md          – new: narration script for install walkthrough video
index.html                            – keyboard shortcuts for approve/reject; CSS for kbd hints
tests/test_differentiation.py         – new: 96 tests covering all 15 Section C items
```

---

## Section C implementation summary

| # | Item | Implementation | Files |
|---|---|---|---|
| C1 | README opening a buyer can scan in one minute | "At a glance" section with five key points at top of README | `README.md` |
| C2 | Text architecture diagram | Exists at `docs/diagrams/architecture.txt`; last-updated and migration count corrected | `docs/diagrams/architecture.txt` |
| C3 | Designed-versus-verified table | Exists in README; verified by test | `README.md` |
| C4 | External-links section | Exists in README with disclaimer; verified by test | `README.md` |
| C5 | File map with one responsibility per package | Exists in README; updated to include `app/domain/buyer_features` and Phase 8 docs | `README.md` |
| C6 | Release timeline in docs/releases | New file listing all 1.0.0 phases and all follow-up phases | `docs/releases/timeline.md` |
| C7 | Comparison page against public desks | New file citing HelpScout, Front, Freshdesk with honesty note; no invented benchmarks | `docs/COMPARISON.md` |
| C8 | Empty states and error copy on the desk | All sections have empty-state messages; login and network error copy verified by test | `index.html` |
| C9 | Keyboard path for approve and reject | `Alt+A` approve, `Alt+R` reject, `↑↓` row focus; hint text visible in Approval section | `index.html` |
| C10 | Consistent names Amin, Amilos, Leila | Confirmed consistent in all agent modules, pipeline, and README | verified by test |
| C11 | Test-house index | New file with run index, honesty rule, external-test section, test file index | `docs/TEST_HOUSE.md` |
| C12 | Install video script | New file — markdown narration script, not a video file; no real secrets | `docs/INSTALL_VIDEO_SCRIPT.md` |
| C13 | Threat model linked from README | Exists in Author section of README; verified by test | `README.md` |
| C14 | No secret in any sample | `.env.example` uses placeholder values only; verified by regex test | verified by test |
| C15 | Final wording that sale and hosting are not done | "What is not claimed" section in README and reiterated in COMPARISON.md | `README.md`, `docs/COMPARISON.md` |

---

## Keyboard shortcuts added (C9)

The Approval section of the desk now supports:

| Key | Action |
|---|---|
| `↑` / `↓` | Move focus between approval rows |
| `Alt+A` | Approve the focused row (or first row if none focused) |
| `Alt+R` | Reject the focused row (or first row if none focused) |

Focus is shown with a blue outline. The shortcuts are active only when the
Approval section is visible. A keyboard hint is displayed above the approval
table.

---

## Test command

```bash
python3 -m pytest tests/ -q
```

**Result:** 600 passed, 0 failed, 0 errors.

Previous baseline: 504 tests (Phase 7 follow-up).  
New tests this phase: 96 (in `tests/test_differentiation.py`).

---

## Requirements satisfied

| Requirement | Satisfied | Notes |
|---|---|---|
| All 15 Section C items implemented | yes | See table above |
| Each item has a passing test | yes | `tests/test_differentiation.py` — 96 tests, all green |
| Comparison notes cite real public products | yes | HelpScout, Front, Freshdesk with public URLs |
| Comparison does not invent benchmarks | yes | Explicit honesty note; regex test guards against numeric throughput claims |
| App stays runnable | yes | 600 tests pass; no existing test broken |
| No private links in any new file | yes | All links are public URLs or internal file paths |
| No secret in any sample | yes | `.env.example` uses placeholders only |

---

## Files changed

| File | Change |
|---|---|
| `README.md` | Added at-a-glance section; updated file map, test count, roadmap status, author links |
| `docs/diagrams/architecture.txt` | Updated last-updated line and migration count (0001–0010) |
| `docs/COMPARISON.md` | **New.** Comparison against HelpScout, Front, Freshdesk |
| `docs/releases/timeline.md` | **New.** Full release timeline |
| `docs/TEST_HOUSE.md` | **New.** Test-house index with run records |
| `docs/INSTALL_VIDEO_SCRIPT.md` | **New.** Narration script for install video |
| `index.html` | Added keyboard shortcuts (C9), CSS for kbd hint and notice-loading classes |
| `tests/test_differentiation.py` | **New.** 96 tests covering all 15 Section C items |
| `docs/releases/follow-up-phase-8.md` | **New.** This file |

---

## Known limitations (not failures)

- C11 test-house index: historical run results for Phases 1–5 are not available
  with exact counts; they are recorded as "passed" without a count.
- C7 comparison: product feature sets are based on public documentation at
  2026-10-09; vendor features may change after this date.
- C9 keyboard shortcuts: `Alt+A`/`Alt+R` may conflict with browser or OS shortcuts
  on some platforms. The operator can use the on-screen Approve and Reject buttons
  as a fallback.

---

## Test failures

None. All 600 tests pass.
