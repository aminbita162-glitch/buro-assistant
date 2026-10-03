"""
app/ingest/providers/base.py – abstract MailProvider interface (row 5/6).

Every provider adapter must implement this interface so the ingest
orchestrator is provider-neutral.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from app.ingest.normalize import NormalizedMessage


class MailProvider(ABC):
    """Abstract base for all mailbox provider adapters."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier stored on every Message row, e.g. 'imap'."""

    @abstractmethod
    def fetch_new(self, tenant_id: int) -> Iterator[NormalizedMessage]:
        """
        Yield NormalizedMessage objects for messages not yet seen.

        Implementations must not mark messages as read / delete them here;
        that is the responsibility of the ingest orchestrator after the
        message is committed to the store.
        """
