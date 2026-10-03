"""
app/ingest/normalize.py – provider-neutral normalised message (row 6).

NormalizedMessage is the single contract between any mail provider adapter
and the ingest orchestrator.  It carries only the fields the rest of the
system needs; all provider-specific raw data lives in Message.raw_json.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Attachment:
    filename: str
    content_type: str
    size_bytes: int


@dataclass
class NormalizedMessage:
    """Provider-neutral message envelope."""

    # Routing
    provider: str               # "imap", "fake", …
    provider_message_id: str    # unique within provider+tenant
    tenant_id: int

    # Headers
    message_id_header: Optional[str]   # RFC 5322 Message-Id value
    subject: str
    subject_normalized: str            # lower-cased, whitespace-collapsed
    sender: str
    recipients: List[str]

    # Body
    body_text: str

    # Attachments (row 12 – populated by provider adapter)
    attachments: List[Attachment] = field(default_factory=list)

    # Raw envelope as a JSON-serialisable dict (row 4).
    raw: dict = field(default_factory=dict)

    @staticmethod
    def normalize_subject(subject: str) -> str:
        """Collapse whitespace and lower-case for duplicate detection."""
        import re
        return re.sub(r"\s+", " ", subject.strip().lower())
