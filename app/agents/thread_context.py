"""
app/agents/thread_context.py – bounded thread context for Amin's prompt (Phase 2).

Fetches the last three messages in the same thread (matched by
subject_normalized per tenant), redacts PII from each body, and returns a
single clip capped at THREAD_CLIP_CHARS total characters.

Rules
-----
- Only messages with state "new" (not duplicate / quarantine) are included.
- The current message is excluded (it is already in the prompt).
- Attachment bytes are never included — body_text from raw_json only.
- A rule hit still costs zero tokens; the clip is passed but not model-called.
- Clip total is capped at THREAD_CLIP_CHARS (200) after redaction.
"""
from __future__ import annotations

import json
from typing import List, Optional

from sqlalchemy.orm import Session

from app.agents.redact import redact
from app.ingest.models import Message

# Maximum total characters for the combined thread context clip.
THREAD_CLIP_CHARS = 200

# Number of prior messages to include.
THREAD_LOOKBACK = 3


def fetch_thread_context(
    db: Session,
    tenant_id: int,
    subject_normalized: str,
    exclude_provider_message_id: str,
) -> str:
    """
    Return a redacted, capped clip of the last THREAD_LOOKBACK messages in
    the same thread (same tenant + subject_normalized) excluding the current
    message.

    Returns an empty string when there are no prior messages.
    """
    rows: List[Message] = (
        db.query(Message)
        .filter(
            Message.tenant_id == tenant_id,
            Message.subject_normalized == subject_normalized,
            Message.provider_message_id != exclude_provider_message_id,
            Message.state == "new",
        )
        .order_by(Message.ingest_time.desc())
        .limit(THREAD_LOOKBACK)
        .all()
    )

    if not rows:
        return ""

    # Rows are newest-first; reverse so oldest is first in the clip.
    rows = list(reversed(rows))

    SEP = " | "
    parts: List[str] = []
    total = 0

    for row in rows:
        if total >= THREAD_CLIP_CHARS:
            break

        body = _extract_body(row)
        if not body:
            continue

        redacted_body, _ = redact(body)
        # Budget must account for the separator that will be added between parts.
        sep_cost = len(SEP) if parts else 0
        remaining = THREAD_CLIP_CHARS - total - sep_cost
        if remaining <= 0:
            break
        snippet = redacted_body[:remaining]
        parts.append(snippet)
        total += sep_cost + len(snippet)

    return SEP.join(parts) if parts else ""


def _extract_body(row: Message) -> str:
    """Extract body_text from the stored raw_json. Attachment bytes are not returned."""
    try:
        raw = json.loads(row.raw_json)
        return str(raw.get("body_text", ""))
    except Exception:  # noqa: BLE001
        return ""
