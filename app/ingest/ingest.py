"""
app/ingest/ingest.py – ingest orchestrator.

Rows closed:
  3  – idempotency: returns existing Message on duplicate provider key
  4  – raw_json written once and never updated
  6  – accepts NormalizedMessage; provider-neutral
  9  – duplicate detection by message_id_header + subject_normalized per tenant
  12 – attachment_state set from provider adapter classification
  Phase 1 – sender authentication: auth_spf / auth_dkim / auth_dmarc stored
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Tuple

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.ingest.models import Message
from app.ingest.normalize import NormalizedMessage
from app.ingest.sender_auth import check_sender_auth


# Result codes returned by ingest_message.
RESULT_NEW = "new"
RESULT_DUPLICATE = "duplicate"
RESULT_QUARANTINE = "quarantine"


def _attachment_state(msg: NormalizedMessage) -> str:
    """Derive attachment_state from the NormalizedMessage attachments list."""
    if not msg.attachments:
        return "none"
    from app.ingest.providers.imap_provider import ATTACHMENT_ALLOWLIST
    for att in msg.attachments:
        if att.content_type.lower() not in ATTACHMENT_ALLOWLIST:
            return "quarantine"
    return "clean"


def ingest_message(
    db: Session,
    msg: NormalizedMessage,
) -> Tuple[Message, str]:
    """
    Store a NormalizedMessage and return (Message, result_code).

    result_code is one of: "new", "duplicate", "quarantine".

    Idempotency (row 3):
        If a row with (tenant_id, provider_message_id) already exists,
        the existing row is returned with result "duplicate".

    Duplicate detection (row 9):
        If message_id_header + subject_normalized match an existing row
        for this tenant, the new row is stored but its state is set to
        "duplicate".

    Attachment quarantine (row 12):
        If any attachment has a disallowed content-type, state is set to
        "quarantine" (overrides duplicate).

    Raw store (row 4):
        raw_json is serialised from msg.raw once and never updated.
    """
    # ---- Idempotency check (row 3) ----
    existing = (
        db.query(Message)
        .filter(
            Message.tenant_id == msg.tenant_id,
            Message.provider_message_id == msg.provider_message_id,
        )
        .first()
    )
    if existing:
        return existing, RESULT_DUPLICATE

    # ---- Determine state ----
    att_state = _attachment_state(msg)

    if att_state == "quarantine":
        state = RESULT_QUARANTINE
    else:
        # Duplicate detection by Message-Id header + normalised subject (row 9).
        duplicate = _find_content_duplicate(db, msg)
        state = RESULT_DUPLICATE if duplicate else RESULT_NEW

    # ---- Sender authentication (Phase 1) ----
    # Run the check using whatever results the message already carries
    # (a provider may pre-populate them) or perform the DNS lookup now.
    # Pre-populated values on msg take precedence so tests can inject results.
    auth_spf = msg.auth_spf
    auth_dkim = msg.auth_dkim
    auth_dmarc = msg.auth_dmarc
    if auth_spf == "not_run" and auth_dkim == "not_run" and auth_dmarc == "not_run":
        # No pre-populated result — run the DNS check.
        raw_headers = msg.raw.get("headers", {}) if isinstance(msg.raw, dict) else {}
        auth_result = check_sender_auth(msg.sender, raw_headers)
        auth_spf = auth_result.spf
        auth_dkim = auth_result.dkim
        auth_dmarc = auth_result.dmarc

    # ---- Store (row 4 – raw_json written once) ----
    record = Message(
        tenant_id=msg.tenant_id,
        provider=msg.provider,
        provider_message_id=msg.provider_message_id,
        raw_json=json.dumps(msg.raw),          # immutable; never updated
        message_id_header=msg.message_id_header,
        subject_normalized=msg.subject_normalized,
        ingest_time=datetime.now(timezone.utc),
        state=state,
        attachment_state=att_state,
        auth_spf=auth_spf,
        auth_dkim=auth_dkim,
        auth_dmarc=auth_dmarc,
    )
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
    except IntegrityError:
        db.rollback()
        # Race: another worker inserted first — idempotency path.
        existing = (
            db.query(Message)
            .filter(
                Message.tenant_id == msg.tenant_id,
                Message.provider_message_id == msg.provider_message_id,
            )
            .first()
        )
        if existing:
            return existing, RESULT_DUPLICATE
        raise

    return record, state


def _find_content_duplicate(db: Session, msg: NormalizedMessage) -> bool:
    """
    Return True if a non-duplicate message with the same Message-Id header
    OR the same normalised subject already exists for this tenant (row 9).
    """
    q = db.query(Message).filter(
        Message.tenant_id == msg.tenant_id,
        Message.state != RESULT_DUPLICATE,
    )

    if msg.message_id_header:
        match = q.filter(
            Message.message_id_header == msg.message_id_header
        ).first()
        if match:
            return True

    if msg.subject_normalized:
        match = q.filter(
            Message.subject_normalized == msg.subject_normalized
        ).first()
        if match:
            return True

    return False
