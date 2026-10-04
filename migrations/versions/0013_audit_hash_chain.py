"""Phase 5 – audit hash chain: add prev_hash column to audit_log

Revision ID: 0013
Revises: 0012
Create Date: 2026-10-21 00:00:00.000000

Adds a nullable String(64) column `prev_hash` to the `audit_log` table.
Existing rows will have NULL in this column; they are treated as pre-chain
rows and are excluded from chain verification.  New rows written by
log_event() will always carry a SHA-256 hex value.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: Union[str, None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if "audit_log" not in insp.get_table_names():
        return

    existing_cols = {c["name"] for c in insp.get_columns("audit_log")}

    if "prev_hash" not in existing_cols:
        op.add_column(
            "audit_log",
            sa.Column("prev_hash", sa.String(64), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if "audit_log" not in insp.get_table_names():
        return

    existing_cols = {c["name"] for c in insp.get_columns("audit_log")}

    if "prev_hash" in existing_cols:
        with op.batch_alter_table("audit_log") as batch_op:
            batch_op.drop_column("prev_hash")
