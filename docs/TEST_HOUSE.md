# Buro Assistant — Test House

**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab
**Last updated:** Phase 6 (Desk signals)

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
| 11 | 2026-10-11 | Amin Azimi | `python3 -m pytest tests/ -q` (Follow-up Phase 10) | 600 passed, 0 failed | External test script written; roadmap frozen — no new tests added this phase |
| 12 | 2026-10-12 | Amin Azimi | `python3 -m pytest tests/ -q` (Commercial Phase 1) | 620 passed, 0 failed | Tenant subscription record — 20 new tests |
| 13 | 2026-10-13 | Amin Azimi | `python3 -m pytest tests/ -q` (Commercial Phase 2) | 635 passed, 0 failed | Gateway boundary — 15 new tests |
| 14 | 2026-10-14 | Amin Azimi | `python3 -m pytest tests/ -q` (Commercial Phase 3) | 670 passed, 0 failed | Plan capability enforcement — 35 new tests |
| 15 | 2026-10-15 | Amin Azimi | `python3 -m pytest tests/ -q` (Commercial Phase 4) | 691 passed, 0 failed | Operator surface — 21 new tests |
| 16 | 2026-10-16 | Amin Azimi | `python3 -m pytest tests/ -q` (Commercial Phase 5) | 691 passed, 0 failed | Record and freeze — no new tests added this phase |
| 17 | 2026-10-17 | Amin Azizi | `python3 -m pytest tests/ -q` (Phase 1 — sender auth) | 725 passed, 0 failed | Sender authentication — 34 new tests |
| 18 | 2026-10-18 | Amin Azizi | `python3 -m pytest tests/ -q` (Phase 2 — bounded thread context) | 747 passed, 0 failed | Bounded thread context — 22 new tests |
| 19 | 2026-10-19 | Amin Azizi | `python3 -m pytest tests/ -q` (Phase 3 — local model route) | 762 passed, 0 failed | Local model route — 15 new tests |
| 20 | 2026-10-20 | Amin Azimi | `python3 -m pytest tests/ -q` (Phase 4 — OAuth mailbox boundary) | 779 passed, 0 failed | OAuth mailbox boundary — 17 new tests |
| 21 | 2026-10-21 | Amin Azimi | `python3 -m pytest tests/ -q` (Phase 5 — audit hash chain) | 797 passed, 0 failed | Audit hash chain — 18 new tests |
| 22 | 2026-10-22 | Amin Azimi | `python3 -m pytest tests/ -q` (Phase 6 — desk signals) | 842 passed, 0 failed | Desk signals — 45 new tests |

---

## External test runs

| # | Date | Actor | Action | Result | Notes |
|---|---|---|---|---|---|
| — | not run | — | `bash docs/EXTERNAL_TEST.sh` | not run | Script written in Follow-up Phase 10. When a second account runs it, this row is replaced with the actual date, actor, and result. |

The claim "a second account has installed and verified this" is not made until
this row is updated with a recorded result.

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
| `docs/EXTERNAL_TEST.sh` | External install-and-verify script (not a test file; run manually) | Follow-up 10 |
| `tests/test_subscription.py` | Tenant subscription states: trial, active, expired, cancelled | Commercial Phase 1 |
| `tests/test_gateway.py` | Payment port: fake adapter, live adapter (secret absent), factory selection | Commercial Phase 2 |
| `tests/test_plan_rules.py` | Plan capability enforcement: desk/mail/agents/trial refusals, token cap, send block | Commercial Phase 3 |
| `tests/test_operator_surface.py` | Operator desk plan surface: /desk/plan, choose-plan state, pipeline capability gating | Commercial Phase 4 |
| `tests/test_sender_auth.py` | Sender auth: SenderAuthResult, check_sender_auth (pass/fail/not_run), ingest storage, pipeline send-block | Phase 1 |
| `tests/test_thread_context.py` | Thread context: fetch, 200-char cap, PII redaction, tenant isolation, rule-hit zero tokens, prompt version | Phase 2 |
| `tests/test_local_model.py` | Local model flag: is_local_model_enabled, FakeLocalModel client, pipeline routing for both flag states, rule-hit zero-model path | Phase 3 |
| `tests/test_oauth_mailbox.py` | OAuth mailbox port: OAuthMailboxPort interface, FakeOAuthProvider, factory selection (tokens absent → fake, partial env → fake, app still runs) | Phase 4 |
| `tests/test_audit_hash_chain.py` | Audit hash chain: prev_hash write, ZERO_HASH sentinel, verify_chain (intact and broken), pre-chain NULL rows excluded, tenant isolation | Phase 5 |
| `tests/test_desk_signals.py` | Desk signals: daily digest shadow, semantic duplicate flag, attachment text extraction, dissatisfied-tone flag routing to Leila | Phase 6 |

---

*For questions contact: Amin Azimi, AI Architect, Azimi Innovation Lab.*
