"""
app/policy/shadow.py – shadow mode: store draft, do not send (row 48).

When a tenant has ``shadow_mode: true`` in its policy config, the pipeline
stores the reply draft in the ``drafts`` table but never calls the mail
sender.  This lets operators validate the system's decisions against real
mail before enabling live auto-reply.

The Draft ORM model and the shadow helper live here.
Migration 0004 creates the ``drafts`` table.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Session

from app.main import Base
from app.policy.send_decision import (
    SEND_DECISION_SHADOW,
    SEND_DECISION_DRAFT,
    should_send,
)


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class Draft(Base):
    __tablename__ = "drafts"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=True, index=True)

    subject = Column(String, nullable=False, default="")
    body = Column(Text, nullable=False, default="")
    template_id = Column(String, nullable=True)
    language = Column(String, nullable=True)
    decision_hash = Column(String, nullable=True)

    # State: draft | shadow | sent | approved | rejected
    state = Column(String, nullable=False, default="draft")

    created_at = Column(DateTime(timezone=True), nullable=False)


# ---------------------------------------------------------------------------
# Shadow store helper (row 48)
# ---------------------------------------------------------------------------

def store_draft(
    db: Session,
    tenant_id: int,
    subject: str,
    body: str,
    template_id: Optional[str] = None,
    language: str = "en",
    decision_hash: Optional[str] = None,
    message_id: Optional[int] = None,
    policy_config: Optional[Dict[str, Any]] = None,
) -> Draft:
    """
    Persist a reply draft.

    The ``state`` field reflects the send decision:
    - ``"shadow"``  → shadow mode is on; draft stored, not sent.
    - ``"draft"``   → auto-reply is off; stored for human review.

    In neither case is the message sent.  Actual sending (state → "sent")
    is handled by the worker layer (Phase 8), which may only proceed when
    ``should_send(policy_config) == "allow"``.
    """
    send_dec = should_send(policy_config)
    state = send_dec if send_dec in (SEND_DECISION_SHADOW, SEND_DECISION_DRAFT) else "draft"

    draft = Draft(
        tenant_id=tenant_id,
        message_id=message_id,
        subject=subject,
        body=body,
        template_id=template_id,
        language=language,
        decision_hash=decision_hash,
        state=state,
        created_at=datetime.now(timezone.utc),
    )
    db.add(draft)
    db.commit()
    db.refresh(draft)
    return draft


def is_shadow_mode(policy_config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True when shadow mode is active for the tenant."""
    return should_send(policy_config) == SEND_DECISION_SHADOW
