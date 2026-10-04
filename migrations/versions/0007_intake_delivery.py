"""Follow-up Phase 2 – intake loop and webhook delivery log

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-05 00:00:00.000000

Adds:
  delivery_log – one row per webhook delivery attempt (signed payloads,
                 retries, success/failure, http_status).
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing_tables = insp.get_table_names()

    if "delivery_log" not in existing_tables:
        op.create_table(
            "delivery_log",
            sa.Column("id",              sa.Integer, primary_key=True),
            sa.Column("tenant_id",       sa.Integer, sa.ForeignKey("tenants.id"), nullable=False),
            sa.Column("subscription_id", sa.Integer,
                      sa.ForeignKey("webhook_subscriptions.id"), nullable=False),
            sa.Column("event_type",      sa.String,  nullable=False),
            sa.Column("attempt",         sa.Integer, nullable=False, server_default="1"),
            sa.Column("status",          sa.String,  nullable=False, server_default="ok"),
            sa.Column("http_status",     sa.Integer, nullable=True),
            sa.Column("error_detail",    sa.Text,    nullable=True),
            sa.Column("payload_preview", sa.Text,    nullable=True),
            sa.Column("delivered_at",    sa.DateTime(timezone=True), nullable=False),
            sa.Column("success",         sa.Boolean, nullable=False, server_default="0"),
        )
        op.create_index("ix_delivery_log_id",              "delivery_log", ["id"])
        op.create_index("ix_delivery_log_tenant_id",       "delivery_log", ["tenant_id"])
        op.create_index("ix_delivery_log_subscription_id", "delivery_log", ["subscription_id"])


def downgrade() -> None:
    op.drop_table("delivery_log")
