# Final Phase 4 — Mailbox OAuth Boundary

**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab
**Date:** 2026-10-20
**Branch:** main
**Contract:** DIRECTIVE.txt — Phase 4

---

## Requirement

> Add an OAuth mailbox port beside the IMAP adapter. Tokens are read from the
> environment only. If absent, the existing fake provider is used and the app
> still runs. No token value in the repo. Tests use the fake port only.

---

## What was built

### OAuth mailbox port — `OAuthMailboxPort`

[`app/ingest/providers/oauth_port.py`](../../app/ingest/providers/oauth_port.py)
defines the abstract `OAuthMailboxPort` interface. It mirrors the existing
`MailProvider` base class but is dedicated to OAuth-authenticated mailboxes.
A production operator wires in a live adapter (Microsoft Graph, Gmail API,
etc.) that satisfies the `provider_name` / `fetch_new` contract. No live
adapter is shipped in this repository.

### Fake OAuth provider — `FakeOAuthProvider`

[`app/ingest/providers/fake_oauth_provider.py`](../../app/ingest/providers/fake_oauth_provider.py)
is an in-memory stub that implements `OAuthMailboxPort`. It accepts
pre-queued `NormalizedMessage` objects and yields them on `fetch_new`. It
contacts no external service, reads no environment variable, and requires no
token. It is the only OAuth provider used in tests.

### Factory — `get_oauth_provider`

[`app/ingest/providers/oauth_factory.py`](../../app/ingest/providers/oauth_factory.py)
inspects three environment variables:

| Variable | Purpose |
|---|---|
| `OAUTH_TOKEN_URL` | Token endpoint URL |
| `OAUTH_CLIENT_ID` | Client identifier |
| `OAUTH_CLIENT_SECRET` | Client secret |

When all three are set and non-empty, the factory is the hook point for a live
adapter (not yet shipped). When any variable is absent or empty, `FakeOAuthProvider`
is returned and the app still runs. No token value is read from or written to
any file in this repository.

---

## Files changed

| File | Change |
|------|--------|
| `app/ingest/providers/oauth_port.py` | New — `OAuthMailboxPort` abstract interface |
| `app/ingest/providers/fake_oauth_provider.py` | New — `FakeOAuthProvider` in-memory stub |
| `app/ingest/providers/oauth_factory.py` | New — `get_oauth_provider` factory |
| `tests/test_oauth_mailbox.py` | New — 17 tests covering port, fake provider, and factory |
| `docs/TEST_HOUSE.md` | Run 20 recorded |

---

## Test run

```
python3 -m pytest tests/test_oauth_mailbox.py -v
17 passed, 0 failed
```

Full suite:

```
python3 -m pytest tests/ -q
779 passed, 0 failed
```

Recorded in `docs/TEST_HOUSE.md` as run 20.

---

## No token value in the repository

No OAuth token, client secret, client identifier, or token URL is stored in
any file in this repository. The factory reads environment variables at
runtime only. Tests use the fake provider exclusively and never require any
credential.

---

## Commercialization status

Commercialization (real payment vendor, market launch) is not done and is not
started in this phase.

---

*For questions contact: Amin Azimi, AI Architect, Azimi Innovation Lab.*
