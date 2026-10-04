"""Follow-up Phase 3 – add department column to messages

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-06 00:00:00.000000

Adds:
  messages.department – stored department field (nullable String).
  Previously, department was inferred at query time from the subject keyword.
  After this migration it is a first-class stored field that can be set at
  ingest time and filtered on directly.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if "messages" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("messages")}
        if "department" not in cols:
            op.add_column(
                "messages",
                sa.Column("department", sa.String, nullable=True),
            )
            op.create_index(
                "ix_messages_department",
                "messages",
                ["department"],
            )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if "messages" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("messages")}
        if "department" in cols:
            op.drop_index("ix_messages_department", table_name="messages")
            with op.batch_alter_table("messages") as batch_op:
                batch_op.drop_column("department")
