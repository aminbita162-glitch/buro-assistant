"""Commercial Phase 1 – tenant subscription record

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-12 00:00:00.000000

Creates:
  tenant_subscriptions — plan_code, status, trial_end, token_cap, tokens_used
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: Union[str, None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing_tables = set(insp.get_table_names())

    if "tenant_subscriptions" not in existing_tables:
        op.create_table(
            "tenant_subscriptions",
            sa.Column("id", sa.Integer, primary_key=True, index=True),
            sa.Column(
                "tenant_id",
                sa.Integer,
                sa.ForeignKey("tenants.id"),
                nullable=False,
                unique=True,
                index=True,
            ),
            sa.Column("plan_code", sa.String, nullable=False, server_default="trial"),
            sa.Column("status", sa.String, nullable=False, server_default="trial"),
            sa.Column("trial_end", sa.DateTime(timezone=True), nullable=True),
            sa.Column("token_cap", sa.Integer, nullable=False, server_default="50000"),
            sa.Column("tokens_used", sa.Integer, nullable=False, server_default="0"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if "tenant_subscriptions" in insp.get_table_names():
        op.drop_table("tenant_subscriptions")
