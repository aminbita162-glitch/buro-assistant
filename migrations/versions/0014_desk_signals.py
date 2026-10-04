"""
migrations/versions/0014_desk_signals.py – Phase 6 desk signals.

Adds two nullable boolean columns to the ``messages`` table:
  - ``semantic_duplicate``  (default False)
  - ``dissatisfied_tone``   (default False)

Existing rows receive the default value of False.
Downgrade drops both columns (batch alter for SQLite compatibility).
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = sa.inspect(op.get_bind()).get_table_names()
    if "messages" not in existing:
        return

    cols = {
        col["name"]
        for col in sa.inspect(op.get_bind()).get_columns("messages")
    }

    with op.batch_alter_table("messages") as batch_op:
        if "semantic_duplicate" not in cols:
            batch_op.add_column(
                sa.Column(
                    "semantic_duplicate",
                    sa.Boolean(),
                    nullable=False,
                    server_default=sa.false(),
                )
            )
        if "dissatisfied_tone" not in cols:
            batch_op.add_column(
                sa.Column(
                    "dissatisfied_tone",
                    sa.Boolean(),
                    nullable=False,
                    server_default=sa.false(),
                )
            )


def downgrade() -> None:
    existing = sa.inspect(op.get_bind()).get_table_names()
    if "messages" not in existing:
        return

    cols = {
        col["name"]
        for col in sa.inspect(op.get_bind()).get_columns("messages")
    }

    with op.batch_alter_table("messages") as batch_op:
        if "dissatisfied_tone" in cols:
            batch_op.drop_column("dissatisfied_tone")
        if "semantic_duplicate" in cols:
            batch_op.drop_column("semantic_duplicate")
