"""Phase 8 – workers: traces, work_queue, dead_letter, quotas; cost columns on decisions

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-03 00:00:00.000000

Rows closed:
  25 – cost_usd, tokens_in, tokens_out added to decisions table
  26 – quotas table (per-tenant daily token budget)
  27 – work_queue table with queue_depth_cap (backpressure enforced in app)
  28 – dead_letter table (DLQ + replay)
  29 – work_queue.lane column (priority lanes: critical=0 … low=3)
  30 – traces table (ingest | decide | draft | send)
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing_tables = insp.get_table_names()

    # ------------------------------------------------------------------
    # Row 25 – cost columns on decisions
    # ------------------------------------------------------------------
    if "decisions" in existing_tables:
        existing_cols = {c["name"] for c in insp.get_columns("decisions")}
        if "tokens_in" not in existing_cols:
            op.add_column("decisions", sa.Column("tokens_in",  sa.Integer, nullable=True))
        if "tokens_out" not in existing_cols:
            op.add_column("decisions", sa.Column("tokens_out", sa.Integer, nullable=True))
        if "cost_usd" not in existing_cols:
            op.add_column("decisions", sa.Column("cost_usd",   sa.Float,   nullable=True))

    # ------------------------------------------------------------------
    # Row 30 – traces table
    # ------------------------------------------------------------------
    if "traces" not in existing_tables:
        op.create_table(
            "traces",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id"), nullable=False),
            sa.Column("message_id", sa.Integer, sa.ForeignKey("messages.id"), nullable=True),
            # stage: ingest | decide | draft | send
            sa.Column("stage", sa.String, nullable=False),
            sa.Column("started_at",  sa.DateTime(timezone=True), nullable=False),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("duration_ms", sa.Float, nullable=True),
            # status: ok | error
            sa.Column("status", sa.String, nullable=False, server_default="ok"),
            sa.Column("detail", sa.Text, nullable=True),
        )
        op.create_index("ix_traces_id",         "traces", ["id"])
        op.create_index("ix_traces_tenant_id",  "traces", ["tenant_id"])
        op.create_index("ix_traces_message_id", "traces", ["message_id"])
        op.create_index("ix_traces_stage",      "traces", ["stage"])

    # ------------------------------------------------------------------
    # Row 26 – quotas table (per-tenant daily token budget)
    # ------------------------------------------------------------------
    if "quotas" not in existing_tables:
        op.create_table(
            "quotas",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("tenant_id",    sa.Integer, sa.ForeignKey("tenants.id"), nullable=False),
            sa.Column("quota_date",   sa.Date,    nullable=False),
            sa.Column("tokens_used",  sa.Integer, nullable=False, server_default="0"),
            sa.Column("cost_usd_used",sa.Float,   nullable=False, server_default="0"),
        )
        op.create_index("ix_quotas_id",        "quotas", ["id"])
        op.create_index("ix_quotas_tenant_id", "quotas", ["tenant_id"])
        op.create_index("ix_quotas_date",      "quotas", ["quota_date"])

    # ------------------------------------------------------------------
    # Rows 27, 29 – work_queue table (priority lanes + backpressure)
    # ------------------------------------------------------------------
    if "work_queue" not in existing_tables:
        op.create_table(
            "work_queue",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("tenant_id",  sa.Integer, sa.ForeignKey("tenants.id"), nullable=False),
            sa.Column("message_id", sa.Integer, sa.ForeignKey("messages.id"), nullable=True),
            # priority: critical | high | medium | low
            sa.Column("priority", sa.String, nullable=False, server_default="medium"),
            # lane: 0=critical 1=high 2=medium 3=low
            sa.Column("lane", sa.Integer, nullable=False, server_default="2"),
            # state: pending | processing | done | failed | dead
            sa.Column("state",      sa.String,              nullable=False, server_default="pending"),
            sa.Column("payload",    sa.Text,                nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_work_queue_id",        "work_queue", ["id"])
        op.create_index("ix_work_queue_tenant_id", "work_queue", ["tenant_id"])
        op.create_index("ix_work_queue_state",     "work_queue", ["state"])
        op.create_index("ix_work_queue_lane",      "work_queue", ["lane"])

    # ------------------------------------------------------------------
    # Row 28 – dead_letter table
    # ------------------------------------------------------------------
    if "dead_letter" not in existing_tables:
        op.create_table(
            "dead_letter",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("tenant_id",        sa.Integer, sa.ForeignKey("tenants.id"),  nullable=False),
            sa.Column("message_id",       sa.Integer, sa.ForeignKey("messages.id"), nullable=True),
            sa.Column("original_item_id", sa.Integer, nullable=True),
            sa.Column("priority",         sa.String,  nullable=False, server_default="medium"),
            sa.Column("payload",          sa.Text,    nullable=True),
            sa.Column("failure_reason",   sa.Text,    nullable=True),
            sa.Column("failed_at",        sa.DateTime(timezone=True), nullable=False),
            sa.Column("replayed_at",      sa.DateTime(timezone=True), nullable=True),
            sa.Column("replay_count",     sa.Integer, nullable=False, server_default="0"),
        )
        op.create_index("ix_dead_letter_id",        "dead_letter", ["id"])
        op.create_index("ix_dead_letter_tenant_id", "dead_letter", ["tenant_id"])


def downgrade() -> None:
    op.drop_table("dead_letter")
    op.drop_table("work_queue")
    op.drop_table("quotas")
    op.drop_table("traces")
    # Note: cost columns on decisions are not removed in downgrade
    # to preserve existing decision data.
