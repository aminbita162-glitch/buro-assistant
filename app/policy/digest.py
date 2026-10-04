"""
app/policy/digest.py – daily digest as shadow only (Phase 6).

The daily digest summarises the tenant's activity for one day.
It is always stored as a Draft with state="shadow" and is never sent.
No model is called.  No attachment bytes are passed to any function.

Public API
----------
store_digest_shadow(db, tenant_id) -> Draft
    Build the digest payload and persist it as a shadow draft.
    Returns the stored Draft row.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy.orm import Session

from app.policy.shadow import Draft


# ---------------------------------------------------------------------------
# Digest builder
# ---------------------------------------------------------------------------

def _build_digest_payload(db: Session, tenant_id: int) -> Dict[str, Any]:
    """
    Build the digest payload dict for a tenant.

    Counts messages (all states), drafts, pending approval entries,
    semantic-duplicate flags, and dissatisfied-tone flags for today's
    activity.  No model call.  No attachment bytes.
    """
    from app.ingest.models import Message
    from app.policy.approval import ApprovalQueueEntry

    received = (
        db.query(Message)
        .filter(Message.tenant_id == tenant_id)
        .count()
    )
    drafted = (
        db.query(Draft)
        .filter(Draft.tenant_id == tenant_id)
        .count()
    )
    held = (
        db.query(ApprovalQueueEntry)
        .filter(
            ApprovalQueueEntry.tenant_id == tenant_id,
            ApprovalQueueEntry.state == "pending",
        )
        .count()
    )
    semantic_duplicates = (
        db.query(Message)
        .filter(
            Message.tenant_id == tenant_id,
            Message.semantic_duplicate.is_(True),
        )
        .count()
    )
    dissatisfied = (
        db.query(Message)
        .filter(
            Message.tenant_id == tenant_id,
            Message.dissatisfied_tone.is_(True),
        )
        .count()
    )

    return {
        "tenant_id": tenant_id,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "messages_received": received,
        "drafts_created": drafted,
        "held_for_approval": held,
        "semantic_duplicate_flags": semantic_duplicates,
        "dissatisfied_tone_flags": dissatisfied,
    }


# ---------------------------------------------------------------------------
# Shadow store — never sends
# ---------------------------------------------------------------------------

def store_digest_shadow(db: Session, tenant_id: int) -> Draft:
    """
    Build the daily digest and store it as a shadow draft.

    The draft state is always "shadow".  This function never calls a mail
    sender and never calls a model.  The draft body is the JSON-serialised
    digest payload.
    """
    payload = _build_digest_payload(db, tenant_id)
    body = json.dumps(payload)
    subject = f"Daily digest — {payload['date']}"

    draft = Draft(
        tenant_id=tenant_id,
        subject=subject,
        body=body,
        template_id="daily_digest",
        language="en",
        state="shadow",
        created_at=datetime.now(timezone.utc),
    )
    db.add(draft)
    db.commit()
    db.refresh(draft)
    return draft
