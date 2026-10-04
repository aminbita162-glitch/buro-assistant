"""Follow-up Phase 4 – privacy pack: legal_hold on messages

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-06 00:00:00.000000

Adds:
  messages.legal_hold – Boolean column (default False).
    When True, retention delete (apply_retention) and right-to-erasure
    (delete_tenant_data) skip this row so legally required evidence is
    preserved until the hold is lifted by an authorised operator.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if "messages" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("messages")}
        if "legal_hold" not in cols:
            op.add_column(
                "messages",
                sa.Column(
                    "legal_hold",
                    sa.Boolean,
                    nullable=False,
                    server_default="0",
                ),
            )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if "messages" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("messages")}
        if "legal_hold" in cols:
            with op.batch_alter_table("messages") as batch_op:
                batch_op.drop_column("legal_hold")
