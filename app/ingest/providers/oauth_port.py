"""
app/ingest/providers/oauth_port.py – OAuth mailbox port (Phase 4).

Provides an abstract OAuth mailbox port beside the existing IMAP adapter.

Design rules (DIRECTIVE.txt Phase 4):
  - Tokens are read from the environment only; no token value is stored
    in this repository.
  - When the required environment variables are absent, the factory falls
    back to the fake provider and the app still runs.
  - No token value appears in this file.

Environment variables (read by the factory, not by the port itself):
  OAUTH_TOKEN_URL   – token endpoint URL (e.g. https://login.example.com/token)
  OAUTH_CLIENT_ID   – client identifier issued by the OAuth provider
  OAUTH_CLIENT_SECRET – client secret issued by the OAuth provider
  OAUTH_MAILBOX     – mailbox identifier / email address to poll
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from app.ingest.normalize import NormalizedMessage


class OAuthMailboxPort(ABC):
    """
    Abstract OAuth mailbox port.

    Concrete adapters implement this interface to fetch messages using
    OAuth 2.0 bearer tokens.  The interface is intentionally minimal so
    that any OAuth-capable mail provider (Microsoft Graph, Google Gmail
    API, etc.) can be wired in without modifying the ingest orchestrator.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier stored on every Message row, e.g. 'oauth'."""

    @abstractmethod
    def fetch_new(self, tenant_id: int) -> Iterator[NormalizedMessage]:
        """
        Yield NormalizedMessage objects for unseen messages.

        Implementations obtain a bearer token from the environment
        (never from a call argument or a hard-coded value) and use it
        to authenticate requests to the mail provider's API.
        """
