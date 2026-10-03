"""
app/agents/redact.py – redaction before any model call (row 13).

Scans body_text and subject for PII patterns and replaces them with
placeholder tokens before the text is sent to any model.

Patterns covered:
  - Email addresses  → [EMAIL]
  - Phone numbers    → [PHONE]
  - Credit-card-like sequences → [CARD]
  - National ID / SSN-like sequences → [ID]
  - IBAN              → [IBAN]
"""
from __future__ import annotations

import re
from typing import Tuple

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_PATTERNS = [
    # Order matters: more-specific first.
    ("IBAN", re.compile(
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}\b",
        re.IGNORECASE,
    )),
    ("CARD", re.compile(
        r"\b(?:\d[ -]?){13,19}\b",
    )),
    ("SSN", re.compile(
        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    )),
    ("PHONE", re.compile(
        r"(?<!\d)(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}(?!\d)",
    )),
    ("EMAIL", re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    )),
]


def redact(text: str) -> Tuple[str, int]:
    """
    Replace PII in *text* with placeholder tokens.

    Returns ``(redacted_text, count)`` where *count* is the total number of
    substitutions made.
    """
    total = 0
    for label, pattern in _PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        total += n
    return text, total


def redact_message(subject: str, body: str) -> Tuple[str, str, int]:
    """
    Redact both subject and body.

    Returns ``(redacted_subject, redacted_body, total_count)``.
    """
    rs, ns = redact(subject)
    rb, nb = redact(body)
    return rs, rb, ns + nb
