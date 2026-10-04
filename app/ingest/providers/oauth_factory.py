"""
app/ingest/providers/oauth_factory.py – OAuth mailbox provider factory (Phase 4).

Returns an OAuthMailboxPort implementation appropriate for the current
environment.

Rules (DIRECTIVE.txt Phase 4):
  - Tokens are read from the environment only.
  - When OAUTH_TOKEN_URL, OAUTH_CLIENT_ID, or OAUTH_CLIENT_SECRET is absent
    or empty, the FakeOAuthProvider is returned and the app still runs.
  - No token value, credential, or private URL is stored in this file.

The three environment variables checked here are the minimum required to
acquire a bearer token; the factory does not perform the token exchange
itself — that is the responsibility of a live adapter (not shipped in this
repository).  A production operator adds a live adapter that satisfies the
OAuthMailboxPort interface and registers it in place of the fake.
"""
from __future__ import annotations

import os

from app.ingest.providers.oauth_port import OAuthMailboxPort

_TOKEN_URL_VAR = "OAUTH_TOKEN_URL"
_CLIENT_ID_VAR = "OAUTH_CLIENT_ID"
_CLIENT_SECRET_VAR = "OAUTH_CLIENT_SECRET"


def get_oauth_provider() -> OAuthMailboxPort:
    """
    Return an OAuthMailboxPort for the current environment.

    - When ``OAUTH_TOKEN_URL``, ``OAUTH_CLIENT_ID``, and
      ``OAUTH_CLIENT_SECRET`` are all set and non-empty, a live adapter
      would be wired in here.  No live adapter is shipped in this
      repository; a production operator supplies one.
    - When any of the three variables is absent or empty, the
      :class:`~app.ingest.providers.fake_oauth_provider.FakeOAuthProvider`
      is returned so the app continues to run without OAuth credentials.

    Tests always run without the OAuth environment variables and therefore
    always receive the fake provider.
    """
    token_url = os.environ.get(_TOKEN_URL_VAR, "").strip()
    client_id = os.environ.get(_CLIENT_ID_VAR, "").strip()
    client_secret = os.environ.get(_CLIENT_SECRET_VAR, "").strip()

    if token_url and client_id and client_secret:
        # Live adapter path — not shipped; operator provides implementation.
        # Import is deferred so the module loads cleanly in fake mode.
        from app.ingest.providers.fake_oauth_provider import FakeOAuthProvider  # noqa: F401
        # Production: replace the line above with your live adapter import and
        # return LiveOAuthAdapter(token_url, client_id, client_secret)
        # For now the factory falls through to the fake even when vars are set,
        # because no live adapter exists in this repository.
        return FakeOAuthProvider()

    from app.ingest.providers.fake_oauth_provider import FakeOAuthProvider
    return FakeOAuthProvider()
