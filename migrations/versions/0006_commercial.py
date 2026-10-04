"""Phase 9 – commercial controls

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-03 00:00:00.000000

Rows closed:
  40 – usage_events table
  41 – export job (no schema change; logic is in app/domain/export.py)
  42 – retention_days column on tenants (default 180)
  43 – sandbox seed command (no schema change)
  44 – api_keys table (scoped, hashed)
  45 – webhook_subscriptions table (signed outbound webhooks)
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing_tables = insp.get_table_names()

    # ------------------------------------------------------------------
    # Row 42 – retention_days on tenants (default 180)
    # ------------------------------------------------------------------
    if "tenants" in existing_tables:
        existing_cols = {c["name"] for c in insp.get_columns("tenants")}
        if "retention_days" not in existing_cols:
            op.add_column(
                "tenants",
                sa.Column("retention_days", sa.Integer, nullable=True,
                          server_default="180"),
            )

    # ------------------------------------------------------------------
    # Row 40 – usage_events table
    # ------------------------------------------------------------------
    if "usage_events" not in existing_tables:
        op.create_table(
            "usage_events",
            sa.Column("id",           sa.Integer, primary_key=True),
            sa.Column("tenant_id",    sa.Integer, sa.ForeignKey("tenants.id"), nullable=False),
            sa.Column("event_type",   sa.String,  nullable=False),
            sa.Column("quantity",     sa.Integer, nullable=False, server_default="1"),
            sa.Column("unit",         sa.String,  nullable=True),
            sa.Column("cost_usd",     sa.Float,   nullable=True),
            sa.Column("actor",        sa.String,  nullable=True),
            sa.Column("reference_id", sa.Integer, nullable=True),
            sa.Column("created_at",   sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_usage_events_id",         "usage_events", ["id"])
        op.create_index("ix_usage_events_tenant_id",  "usage_events", ["tenant_id"])
        op.create_index("ix_usage_events_event_type", "usage_events", ["event_type"])

    # ------------------------------------------------------------------
    # Row 44 – api_keys table
    # ------------------------------------------------------------------
    if "api_keys" not in existing_tables:
        op.create_table(
            "api_keys",
            sa.Column("id",           sa.Integer,  primary_key=True),
            sa.Column("tenant_id",    sa.Integer,  sa.ForeignKey("tenants.id"), nullable=False),
            sa.Column("name",         sa.String,   nullable=False),
            sa.Column("key_hash",     sa.String,   nullable=False, unique=True),
            sa.Column("scope",        sa.String,   nullable=True),
            sa.Column("created_at",   sa.DateTime(timezone=True), nullable=False),
            sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("revoked",      sa.Boolean,  nullable=False, server_default="0"),
        )
        op.create_index("ix_api_keys_id",        "api_keys", ["id"])
        op.create_index("ix_api_keys_tenant_id", "api_keys", ["tenant_id"])
        op.create_index("ix_api_keys_key_hash",  "api_keys", ["key_hash"], unique=True)

    # ------------------------------------------------------------------
    # Row 45 – webhook_subscriptions table
    # ------------------------------------------------------------------
    if "webhook_subscriptions" not in existing_tables:
        op.create_table(
            "webhook_subscriptions",
            sa.Column("id",          sa.Integer, primary_key=True),
            sa.Column("tenant_id",   sa.Integer, sa.ForeignKey("tenants.id"), nullable=False),
            sa.Column("url",         sa.String,  nullable=False),
            sa.Column("secret_hash", sa.String,  nullable=False),
            sa.Column("events",      sa.String,  nullable=False, server_default=""),
            sa.Column("active",      sa.Boolean, nullable=False, server_default="1"),
            sa.Column("created_at",  sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_webhook_subscriptions_id",        "webhook_subscriptions", ["id"])
        op.create_index("ix_webhook_subscriptions_tenant_id", "webhook_subscriptions", ["tenant_id"])


def downgrade() -> None:
    op.drop_table("webhook_subscriptions")
    op.drop_table("api_keys")
    op.drop_table("usage_events")
    # retention_days column left in place on downgrade to preserve tenant data.
