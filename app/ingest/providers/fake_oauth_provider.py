"""
app/ingest/providers/fake_oauth_provider.py – Fake OAuth provider (Phase 4).

In-process stub that satisfies the OAuthMailboxPort interface without
contacting any external service or requiring any token.

Used:
  - directly in tests (the directive requires tests use the fake port only)
  - by the factory when OAUTH_TOKEN_URL / OAUTH_CLIENT_ID /
    OAUTH_CLIENT_SECRET are absent from the environment

No token value, URL, credential, or private link appears in this file.
"""
from __future__ import annotations

from typing import Iterator, List

from app.ingest.normalize import NormalizedMessage
from app.ingest.providers.oauth_port import OAuthMailboxPort


class FakeOAuthProvider(OAuthMailboxPort):
    """
    In-memory fake OAuth mailbox provider for tests.

    Usage::

        provider = FakeOAuthProvider()
        provider.queue(NormalizedMessage(...))
        msgs = list(provider.fetch_new(tenant_id=1))

    The fake provider never contacts any external service, never reads
    any environment variable, and never requires a token.
    """

    def __init__(self) -> None:
        self._queue: List[NormalizedMessage] = []

    @property
    def provider_name(self) -> str:
        return "fake_oauth"

    def queue(self, msg: NormalizedMessage) -> None:
        """Add a message to be returned by the next fetch_new call."""
        self._queue.append(msg)

    def fetch_new(self, tenant_id: int) -> Iterator[NormalizedMessage]:
        """Yield all queued messages and clear the queue."""
        while self._queue:
            msg = self._queue.pop(0)
            msg.tenant_id = tenant_id
            yield msg
