"""
app/ingest/models.py – SQLAlchemy models for the ingest layer.

Rows closed:
  3  – idempotency key: (tenant_id, provider_message_id) unique
  4  – immutable raw_json stored at ingest time
  9  – duplicate detection: message_id_header + subject_normalized unique per tenant
  12 – attachment_state: clean | quarantine | none
  Phase 1 – sender authentication: spf, dkim, dmarc stored per message
  Phase 6 – semantic_duplicate flag (bool); dissatisfied_tone flag (bool)
"""
from __future__ import annotations

from sqlalchemy import (
    Boolean, Column, Integer, String, Text, DateTime, UniqueConstraint, ForeignKey,
)
from sqlalchemy.orm import declarative_base

# Re-use the same Base so Alembic sees all models together.
from app.main import Base  # noqa: E402


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)

    # Idempotency key (row 3): provider + provider_message_id unique per tenant.
    provider = Column(String, nullable=False)          # "imap" | "fake" | …
    provider_message_id = Column(String, nullable=False)

    # Immutable raw store (row 4): JSON-serialised envelope, never updated.
    raw_json = Column(Text, nullable=False)

    # Normalised fields used for duplicate detection (row 9).
    message_id_header = Column(String, nullable=True)  # RFC 5322 Message-Id
    subject_normalized = Column(String, nullable=True)

    # Stored department (Phase 3 – row 24 upgrade).
    # Previously inferred from subject keyword; now a first-class stored field.
    department = Column(String, nullable=True, index=True)

    # Ingest metadata.
    ingest_time = Column(DateTime(timezone=True), nullable=False)
    state = Column(String, nullable=False, default="new")
    # state values: new | duplicate | quarantine

    # Attachment state (row 12).
    attachment_state = Column(String, nullable=False, default="none")
    # attachment_state values: none | clean | quarantine

    # Legal hold (Phase 4 / Section B item 12).
    # When True, retention delete and right-to-erasure skip this row.
    legal_hold = Column(Boolean, nullable=False, default=False)

    # Sender authentication (Phase 1).
    # Values: "pass" | "fail" | "not_run"
    # "not_run" means live DNS was not configured at ingest time.
    auth_spf = Column(String, nullable=False, default="not_run")
    auth_dkim = Column(String, nullable=False, default="not_run")
    auth_dmarc = Column(String, nullable=False, default="not_run")

    # Phase 6 – desk signals.
    # semantic_duplicate: True when body semantics suggest a near-duplicate of
    #   an existing message.  Set as a flag only; no second message is created.
    # dissatisfied_tone: True when the body signals customer dissatisfaction.
    #   When True the pipeline routes to Leila and does not send.
    semantic_duplicate = Column(Boolean, nullable=False, default=False)
    dissatisfied_tone = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        # Row 3 – idempotency key.
        UniqueConstraint(
            "tenant_id", "provider_message_id",
            name="uq_messages_tenant_provider_msg_id",
        ),
    )
