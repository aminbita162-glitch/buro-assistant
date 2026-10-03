"""
app/ingest/providers/fake_provider.py – in-process fake adapter for tests.

Lets tests inject arbitrary messages without a real mail server.
The fake provider never talks to any external system.
"""
from __future__ import annotations

from typing import Iterator, List

from app.ingest.normalize import Attachment, NormalizedMessage
from app.ingest.providers.base import MailProvider


class FakeProvider(MailProvider):
    """
    In-memory mail provider for tests.

    Usage::

        provider = FakeProvider()
        provider.queue(NormalizedMessage(...))
        msgs = list(provider.fetch_new(tenant_id=1))
    """

    def __init__(self) -> None:
        self._queue: List[NormalizedMessage] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    def queue(self, msg: NormalizedMessage) -> None:
        """Add a message to be returned by the next fetch_new call."""
        self._queue.append(msg)

    def fetch_new(self, tenant_id: int) -> Iterator[NormalizedMessage]:
        """Yield all queued messages and clear the queue."""
        while self._queue:
            msg = self._queue.pop(0)
            # Ensure tenant_id is set correctly.
            msg.tenant_id = tenant_id
            yield msg
