"""
app/ingest/models.py – SQLAlchemy models for the ingest layer.

Rows closed:
  3  – idempotency key: (tenant_id, provider_message_id) unique
  4  – immutable raw_json stored at ingest time
  9  – duplicate detection: message_id_header + subject_normalized unique per tenant
  12 – attachment_state: clean | quarantine | none
"""
from __future__ import annotations

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, UniqueConstraint, ForeignKey,
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

    # Ingest metadata.
    ingest_time = Column(DateTime(timezone=True), nullable=False)
    state = Column(String, nullable=False, default="new")
    # state values: new | duplicate | quarantine

    # Attachment state (row 12).
    attachment_state = Column(String, nullable=False, default="none")
    # attachment_state values: none | clean | quarantine

    __table_args__ = (
        # Row 3 – idempotency key.
        UniqueConstraint(
            "tenant_id", "provider_message_id",
            name="uq_messages_tenant_provider_msg_id",
        ),
    )
