# Phase 5 Release Notes

**Product:** Buro Assistant
**Release tag:** phase-5
**Date:** 2026-10-03
**Checklist rows closed:** 7, 8, 10, 11, 13, 14, 15, 46, 47

---

## Summary

Phase 5 delivers the three-agent pipeline: Amin (triage controller), Amilos (reply agent), and Leila (supervisor). A rule engine, redaction layer, language detector, urgency lexicon, and deterministic decision hash support them. Fifty synthetic golden messages and full schema contract tests round out the phase.

---

## What was built

### Three JSON schemas (`schemas/`)

| File | Agent | Rows |
|---|---|---|
| `schemas/triage_decision.json` | Amin | 14, 15, 47 |
| `schemas/reply_draft.json` | Amilos | 14, 15, 47 |
| `schemas/supervisor_decision.json` | Leila | 15, 47 |

Each schema is Draft-07 JSON Schema with a `required` array and `additionalProperties: false`. Every agent validates its own output against its schema before returning (row 15). When `jsonschema` is installed, full validation runs; when it is not, a required-key check is applied.

---

### `app/agents/language.py` — language detection (row 10)

- Calls `langdetect.detect()` when the library is installed (library first).
- Falls back to a built-in character-set heuristic (Arabic script → `fa`; CJK → `zh`; French/German/Spanish word markers) so tests never require an external package.

### `app/agents/urgency.py` — urgency lexicon (row 11)

- Base lexicon of ~20 terms maps phrases → integer scores.
- `classify_urgency()` returns `critical | high | medium | low` based on cumulative score.
- `tenant_overrides` dict merged on top of the base lexicon so tenants can add domain terms or set a base term to 0 to suppress it.

### `app/agents/rules.py` — tenant rule engine (rows 7, 8)

- Rule pack JSON: `domain_rules`, `subject_rules`, `department_rules`.
- First-match evaluation: domain → subject → department.
- When a rule fires, the model client is not called (row 7).
- `get_confidence_threshold()` reads the tenant threshold (default 0.7) for row 8.

### `app/agents/redact.py` — PII redaction (row 13)

- Compiled regex patterns for email, phone, credit card, SSN/NID, IBAN.
- `redact_message(subject, body)` returns `(redacted_subject, redacted_body, count)`.
- Amin calls `redact_message` before building any model prompt; raw PII is never sent to the model.

### `app/agents/decision_hash.py` — deterministic hash (row 14)

- `compute_decision_hash(subject_normalized, rule_hit, action, rule_pack)`.
- Serialises to canonical JSON (`sort_keys=True`) then SHA-256.
- Same inputs with the model disabled produce the same hash.

### `app/agents/fake_model.py` — test support

- `FakeModel(responses=[...])` pops responses in order; `None` sentinel raises `ModelCallError`.
- Records all prompts in `fake_model.calls` for assertion.
- No external service is called.

---

### `app/agents/amin.py` — Amin, triage controller (rows 7, 8, 10, 11, 13, 14, 15)

Pipeline per message:
1. Redact PII from subject + body (row 13).
2. Detect language on redacted body (row 10).
3. Score urgency with tenant overrides (row 11).
4. Evaluate tenant rules (row 7). Rule hit → skip model call.
5. If no rule: call model. Model confidence < threshold → route to Leila (row 8).
6. Attach `schema_version`, `prompt_version`, `decision_hash` (row 14).
7. Validate output against `triage_decision.json` (row 15).

### `app/agents/amilos.py` — Amilos, reply agent (rows 14, 15)

- Accepts a triage decision dict and an approved template string.
- Substitutes `{variable}` placeholders from the policy-supplied variables dict.
- Checks the body for forbidden content (invented prices, legal promises, literal dates) before returning.
- Validates output against `reply_draft.json` (row 15).
- Stores `schema_version`, `prompt_version`, `decision_hash` (row 14).

### `app/agents/leila.py` — Leila, supervisor (row 15)

- Accepts an exception context dict with a `reason` field.
- Deterministic mapping: `low_confidence` → `request_human`; `schema_invalid` → `hold`; `quarantine` → `reject`; `no_department` → `reroute`.
- Allowed actions: `hold`, `request_human`, `reject`, `reroute`. Cannot produce `send` or `delete`.
- Validates output against `supervisor_decision.json` (row 15).

---

### `migrations/versions/0003_decisions.py`

Creates the `decisions` table with `schema_version`, `prompt_version`, `decision_hash`, `agent`, `action`, `department`, `language`, `urgency`, `confidence`, `rule_hit`, `reason`, `message_id` FK, `tenant_id` FK, `created_at`. Applied automatically at startup.

---

### `tests/golden/golden_messages.json` — 50 synthetic messages (row 46)

Fifty records covering: billing, support, HR, general; urgency levels from critical to low; multiple languages (English, Persian, French); attachment types including quarantined EXE. Each record carries `expected_department`, `expected_action`, `expected_urgency`.

### `tests/golden/__init__.py`

`load_golden_messages()` returns the list of 50 dicts.

---

## Tests

| File | Tests | Rows |
|---|---|---|
| `tests/test_agents.py` | 46 | 7, 8, 10, 11, 13, 14, 15, 46 |
| `tests/test_contracts.py` | 24 | 47 |

Full suite: **148 tests, 0 failures**.

---

## Capability matrix delta

| Row | Capability | Status |
|---|---|---|
| 7 | Tenant rule pack for domain, subject, and department | ✓ Phase 5 |
| 8 | Confidence threshold; below threshold goes to Supervisor | ✓ Phase 5 |
| 10 | Language detection by library first | ✓ Phase 5 |
| 11 | Urgency lexicon with tenant overrides | ✓ Phase 5 |
| 13 | Redaction before any model call | ✓ Phase 5 |
| 14 | Prompt and schema version stored on each decision | ✓ Phase 5 |
| 15 | Schema validation of agent output | ✓ Phase 5 |
| 46 | Fifty synthetic golden messages with expected decisions | ✓ Phase 5 |
| 47 | Contract tests for the three schemas | ✓ Phase 5 |

---

## What is not in this phase

- No API endpoints exposing agent decisions (Phase 7).
- No live model calls (Phase 5 uses FakeModel in tests; OpenAI client wired in app/main.py is for the existing /assistant and /analyze routes, not the agent pipeline).
- No send decision or approval queue (Phase 6).
- No worker queue driving the pipeline (Phase 8).
