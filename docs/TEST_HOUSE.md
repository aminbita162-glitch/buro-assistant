# Buro Assistant — Test House

**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab
**Last updated:** Follow-up Phase 9

---

## Purpose

This document is the index of all test runs. Every run is recorded here with
date, actor, action, and result. A failed test is written as failed. No run is
hidden. An entry is not written for a run that was not performed.

---

## Honesty rule

Designed-for is not the same as verified. A capability is only marked verified
when a passing test exists in this repository or an external test is recorded
below. Throughput figures are not written here unless they come from a measured
run.

---

## Test run index

| # | Date | Actor | Action | Result | Notes |
|---|---|---|---|---|---|
| 1 | 2026-10-03 | Amin Azimi | `python3 -m pytest tests/ -q` (Phase 1–10 baseline) | 404 passed, 0 failed | Initial 1.0.0 release test run |
| 2 | 2026-10-04 | Amin Azimi | `python3 -m pytest tests/ -q` (Follow-up Phase 1) | passed | Pipeline wire-up |
| 3 | 2026-10-05 | Amin Azimi | `python3 -m pytest tests/ -q` (Follow-up Phase 2) | passed | Live intake and delivery |
| 4 | 2026-10-06 | Amin Azimi | `python3 -m pytest tests/ -q` (Follow-up Phase 3) | passed | Operator desk |
| 5 | 2026-10-06 | Amin Azimi | `python3 -m pytest tests/ -q` (Follow-up Phase 4) | passed | Privacy pack |
| 6 | 2026-10-07 | Amin Azimi | `python3 -m pytest tests/ -q` (Follow-up Phase 5) | 454 passed, 0 failed | Token policy |
| 7 | 2026-10-08 | Amin Azimi | `python3 -m pytest tests/ -q` (Follow-up Phase 6) | 454 passed, 0 failed | README and release surface |
| 8 | 2026-10-08 | Amin Azimi | `python3 -m pytest tests/ -q` (Follow-up Phase 7) | 504 passed, 0 failed | 50 new buyer-feature tests |
| 9 | 2026-10-09 | Amin Azimi | `python3 -m pytest tests/ -q` (Follow-up Phase 8) | 600 passed, 0 failed | Differentiation pack — 96 new tests |
| 10 | 2026-10-10 | Amin Azimi | `python3 -m pytest tests/ -q` (Follow-up Phase 9) | 600 passed, 0 failed | Test-house record and 1.1.0 release notes — no new tests added this phase |

---

## External test runs

No external test run has been performed. The external test script is planned for
Follow-up Phase 10. Until that run is recorded here with a result, the claim
"a second account can install and verify this" is not made.

---

## Failed tests

No test failures have been hidden. All failed tests from any run would be
recorded here.

---

## Test file index

| File | Covers | First introduced |
|---|---|---|
| `tests/test_security.py` | Security headers, rate limits, health, migrations | Phase 2 |
| `tests/test_tenancy.py` | Tenant isolation, cross-tenant rejection | Phase 3 |
| `tests/test_ingest.py` | Idempotency, raw store, IMAP adapter, normalization | Phase 4 |
| `tests/test_agents.py` | Rule engine, triage, language, urgency, golden messages | Phase 5 |
| `tests/test_contracts.py` | JSON schema contracts for three agent schemas | Phase 5 |
| `tests/test_policy.py` | Templates, send decision, SLA, approval queue, audit, shadow | Phase 6 |
| `tests/test_desk.py` | Dashboard counts, department queues | Phase 7 |
| `tests/test_workers.py` | Queue, quota, backpressure, dead letter, traces, cost | Phase 8 |
| `tests/test_commercial.py` | Usage, export, retention, sandbox, API keys, webhooks | Phase 9 |
| `tests/test_release.py` | Capability matrix, release surface | Phase 10 |
| `tests/test_token_policy.py` | Rule-hit zero tokens, attachment exclusion, triage cache | Follow-up 5 |
| `tests/test_pipeline.py` | End-to-end pipeline integration | Follow-up 1 |
| `tests/test_intake.py` | Intake worker loop with fake adapter | Follow-up 2 |
| `tests/test_privacy.py` | GDPR export, erasure, retention enforcement, legal hold | Follow-up 4 |
| `tests/test_webhook_delivery.py` | Signed webhook delivery, retries, delivery log | Follow-up 2 |
| `tests/test_buyer_features.py` | All 15 Section B buyer options | Follow-up 7 |
| `tests/test_differentiation.py` | All 15 Section C finish items | Follow-up 8 |

---

*For questions contact: Amin Azimi, AI Architect, Azimi Innovation Lab.*
