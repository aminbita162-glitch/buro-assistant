# Phase 7 — Air-gap package

**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab
**Date:** 2026-10-23
**Branch:** main

---

## Summary

Phase 7 adds a Docker Compose file that runs the application and its database
with no cloud model client and no gateway secret. The README states that this
is operator-hosted. Tests verify that the compose file names no secret.

---

## What was built

### `docker-compose.yml`

New file at the repo root. Defines two services:

| Service | Image | Purpose |
|---|---|---|
| `db` | `postgres:16-alpine` | Bundled PostgreSQL database. Data persisted in a named volume (`db_data`). Healthcheck waits for `pg_isready` before the app starts. |
| `app` | Built from `Dockerfile` | Application service. |

Environment variables set in the compose file:

| Variable | Value | Effect |
|---|---|---|
| `DATABASE_URL` | `postgresql://buro:buro@db:5432/buro` | Connects to the bundled `db` service. |
| `PORT` | `8000` | Bind port. |
| `LOCAL_MODEL` | `1` | Disables the cloud model client. `FakeModel` is used for all agent calls. |

Variables intentionally absent from the compose file:

| Variable | Reason absent |
|---|---|
| `OPENAI_API_KEY` | Cloud model client not called when `LOCAL_MODEL=1`. No key needed. |
| `GATEWAY_SECRET` | Absent means the fake payment adapter is used. No gateway call is made. |
| `IMAP_HOST` / `IMAP_USER` / `IMAP_PASSWORD` | Absent means `FakeProvider` is used. No live mailbox is polled. |

A comment block at the top of the file states:

> "Operator-hosted. No cloud model client is called. No gateway secret is required."

The operator desk is exposed on port `8000`.

### `Dockerfile`

New file at the repo root. Python 3.12-slim base image. Installs
`requirements.txt`, copies the source, and starts the app with `uvicorn`.

### `README.md`

New section **Air-gap package (operator-hosted)** added before the existing
Quick start section. States:

- The stack is started with `docker compose up`.
- `LOCAL_MODEL=1` is set so the cloud model client is never called.
- `OPENAI_API_KEY` and `GATEWAY_SECRET` are intentionally absent.
- This is operator-hosted. Azimi Innovation Lab does not operate or monitor
  the stack.

### `tests/test_air_gap.py`

14 new tests across 5 classes:

| Class | Tests | What is verified |
|---|---|---|
| `TestComposeFileExists` | 2 | File exists at repo root; file is not empty |
| `TestNoSecretNamed` | 4 | `OPENAI_API_KEY` not assigned; `GATEWAY_SECRET` not assigned; no `sk-...` pattern present; `IMAP_PASSWORD` not assigned |
| `TestLocalModelEnabled` | 1 | `LOCAL_MODEL: "1"` is present |
| `TestOperatorHostedStatement` | 1 | The word "operator-hosted" appears in the file |
| `TestComposeStructure` | 6 | `services:` block present; `db:` service present; `app:` service present; `volumes:` block present; no host network mode; postgres image referenced |

---

## Tests — `tests/test_air_gap.py`

14 new tests, all passing.

---

## Test run

| # | Date | Actor | Action | Result |
|---|---|---|---|---|
| 23 | 2026-10-23 | Amin Azimi | `python3 -m pytest tests/ -q` | 856 passed, 0 failed |

---

## Honesty note

- The compose file is not tested with a live Docker daemon in this repository.
  The tests verify the compose file's text content — that it names no secret,
  sets `LOCAL_MODEL`, and carries the operator-hosted statement.
- `docker compose up` has not been run as part of this phase. The file is
  designed for operator use; the operator is responsible for verifying the
  container starts correctly in their environment.
- The bundled PostgreSQL credentials (`buro`/`buro`) are for the isolated
  container network only. They are not secrets; they are not used outside
  the compose stack. Operators who expose the database port should change them.

---

## Files changed

| File | Change |
|---|---|
| `docker-compose.yml` | New: operator-hosted stack (app + db, no cloud model, no gateway secret) |
| `Dockerfile` | New: Python 3.12-slim image for the app service |
| `README.md` | Added "Air-gap package (operator-hosted)" section |
| `tests/test_air_gap.py` | 14 new tests — compose file names no secret |
| `docs/TEST_HOUSE.md` | Run 23 recorded |
| `docs/releases/final-phase-7.md` | This file |
