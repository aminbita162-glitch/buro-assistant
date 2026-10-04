"""Phase 1 – sender authentication: auth_spf, auth_dkim, auth_dmarc on messages

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-17 00:00:00.000000

Adds three nullable String columns to the messages table:
  auth_spf   – SPF check result:   "pass" | "fail" | "not_run"
  auth_dkim  – DKIM check result:  "pass" | "fail" | "not_run"
  auth_dmarc – DMARC check result: "pass" | "fail" | "not_run"

Default value is "not_run" for all three so existing rows are unaffected.
A fail does not auto-send and does not delete the message.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: Union[str, None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if "messages" not in insp.get_table_names():
        return

    existing_cols = {c["name"] for c in insp.get_columns("messages")}

    for col_name in ("auth_spf", "auth_dkim", "auth_dmarc"):
        if col_name not in existing_cols:
            op.add_column(
                "messages",
                sa.Column(
                    col_name,
                    sa.String,
                    nullable=False,
                    server_default="not_run",
                ),
            )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if "messages" not in insp.get_table_names():
        return

    existing_cols = {c["name"] for c in insp.get_columns("messages")}

    cols_to_drop = [c for c in ("auth_spf", "auth_dkim", "auth_dmarc") if c in existing_cols]
    if cols_to_drop:
        with op.batch_alter_table("messages") as batch_op:
            for col_name in cols_to_drop:
                batch_op.drop_column(col_name)
