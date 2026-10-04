"""
app/ingest/attachment_text.py – text extraction from allowed attachment types
(Phase 6).

Rules
-----
- Only extract from content-types in ATTACHMENT_ALLOWLIST (already approved
  by the ingest layer — no new allowlist is introduced here).
- Only text/plain and text/csv yield bytes that are decoded to a string.
- application/pdf is in the allowlist but binary parsing is not performed
  here; a placeholder empty string is returned so the interface is uniform.
- All other allowed types (images, docx, xlsx) yield an empty string —
  image pixels and binary office data are not text.
- The extracted string is never passed to a model.  Callers must not forward
  the return value to any model client.
- Attachment bytes never leave this module as anything other than a plain
  Python str.

Public API
----------
extract_text(content_type: str, data: bytes) -> str
    Return the plain-text content of *data* for supported types.
    Returns "" for unsupported or non-text types.
    Never raises; returns "" on decode error.

allowed_for_extraction(content_type: str) -> bool
    Return True when the content-type is in the allowlist.
"""
from __future__ import annotations

from app.ingest.providers.imap_provider import ATTACHMENT_ALLOWLIST

# Content-types from which text bytes are decoded.
_TEXT_TYPES: frozenset[str] = frozenset({"text/plain", "text/csv"})


def allowed_for_extraction(content_type: str) -> bool:
    """Return True when *content_type* is in the attachment allowlist."""
    return content_type.lower() in ATTACHMENT_ALLOWLIST


def extract_text(content_type: str, data: bytes) -> str:
    """
    Return the plain-text content of *data* for supported types.

    Only text/plain and text/csv bytes are decoded to a string.
    application/pdf and binary office formats return "".
    Images return "".
    The returned string must not be forwarded to a model.
    """
    ct = content_type.lower()
    if ct not in ATTACHMENT_ALLOWLIST:
        # Not an allowed type at all — return empty.
        return ""
    if ct in _TEXT_TYPES:
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""
    # PDF and binary office formats: text extraction is not performed.
    # The allowlist permits these types but they are not decoded here.
    return ""
