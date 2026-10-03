"""Phase 4 – ingest: add messages table

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-03 00:00:00.000000

Adds the messages table with:
  - idempotency key  (tenant_id, provider_message_id)  – row 3
  - immutable raw_json column                           – row 4
  - message_id_header, subject_normalized               – row 9
  - attachment_state                                    – row 12
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if "messages" in insp.get_table_names():
        return  # idempotent

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "tenant_id", sa.Integer,
            sa.ForeignKey("tenants.id"), nullable=False,
        ),
        sa.Column("provider", sa.String, nullable=False),
        sa.Column("provider_message_id", sa.String, nullable=False),
        sa.Column("raw_json", sa.Text, nullable=False),
        sa.Column("message_id_header", sa.String, nullable=True),
        sa.Column("subject_normalized", sa.String, nullable=True),
        sa.Column(
            "ingest_time",
            sa.DateTime(timezone=True), nullable=False,
        ),
        sa.Column("state", sa.String, nullable=False, server_default="new"),
        sa.Column(
            "attachment_state", sa.String, nullable=False,
            server_default="none",
        ),
        sa.UniqueConstraint(
            "tenant_id", "provider_message_id",
            name="uq_messages_tenant_provider_msg_id",
        ),
    )
    op.create_index("ix_messages_id", "messages", ["id"])
    op.create_index("ix_messages_tenant_id", "messages", ["tenant_id"])
    op.create_index(
        "ix_messages_tenant_subject",
        "messages",
        ["tenant_id", "subject_normalized"],
    )
    op.create_index(
        "ix_messages_tenant_message_id_header",
        "messages",
        ["tenant_id", "message_id_header"],
    )


def downgrade() -> None:
    op.drop_table("messages")
