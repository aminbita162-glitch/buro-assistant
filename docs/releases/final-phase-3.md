# Final Phase 3 — Local Model Route

**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab
**Date:** 2026-10-20
**Branch:** main
**Contract:** DIRECTIVE.txt — Phase 3

---

## Requirement

> Add a tenant flag for local model. When set, the cloud model client is not
> called. The fake local client is used in tests. No new vendor key in the
> repo. Tests for both flags.

---

## What was built

### Tenant flag — `is_local_model_enabled`

`app/agents/local_model.py` exposes `is_local_model_enabled(policy_config)`.
It returns `True` when the tenant's policy config dict contains
`local_model: true`. A missing key, a `None` config, or an explicit `false`
all return `False`. Existing tenants without the key are unaffected.

### Pipeline routing

`app/pipeline.py` reads the flag at the top of `run_pipeline`. When the flag
is `True`, `effective_triage_model` is set to the `local_model` parameter
instead of `triage_model`. The cloud model client is never called for that
path. A rule hit bypasses both clients regardless of the flag, costing zero
tokens.

### Fake local client — `FakeLocalModel`

`FakeLocalModel` in `app/agents/local_model.py` satisfies the
`.call(prompt) -> dict` contract. It queues canned responses, records prompts
in `.calls`, injects a default confidence when absent, and raises
`LocalModelError` on a `None` sentinel. It is a distinct type from `FakeModel`
so tests can assert which client the pipeline chose. No real inference vendor
and no new vendor key are introduced.

---

## Files changed

| File | Change |
|------|--------|
| `app/agents/local_model.py` | New — `is_local_model_enabled`, `FakeLocalModel`, `LocalModelError` |
| `app/pipeline.py` | Phase 3 routing block (lines 212–218) |
| `tests/test_local_model.py` | 15 new tests covering both flag states and the rule-hit path |

---

## Test run

```
python3 -m pytest tests/test_local_model.py -v
15 passed, 0 failed
```

Full suite:

```
python3 -m pytest tests/ -q
762 passed, 0 failed
```

Recorded in `docs/TEST_HOUSE.md` as run 19.

---

## No vendor key in the repository

`FakeLocalModel` is the only local-model client shipped here. A production
operator replaces it with their own implementation. No token, API key, or
credential for any local-inference service was added to this repository.

---

## Commercialization status

Commercialization (real payment vendor, market launch) is not done and is not
started in this phase.

---

*For questions contact: Amin Azimi, AI Architect, Azimi Innovation Lab.*
