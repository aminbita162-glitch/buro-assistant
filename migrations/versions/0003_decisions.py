"""Phase 5 – agents: add decisions table

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-03 00:00:00.000000

Stores one row per triage or supervisor decision (rows 14, 15):
  - schema_version, prompt_version  – row 14
  - decision_hash                   – row 14 (deterministic hash)
  - agent: "amin" | "leila"
  - action, department, language, urgency, confidence, rule_hit
  - message_id FK → messages
  - tenant_id FK → tenants
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if "decisions" in insp.get_table_names():
        return  # idempotent

    op.create_table(
        "decisions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "tenant_id", sa.Integer,
            sa.ForeignKey("tenants.id"), nullable=False,
        ),
        sa.Column(
            "message_id", sa.Integer,
            sa.ForeignKey("messages.id"), nullable=True,
        ),
        # Row 14 – version fields stored on every decision.
        sa.Column("schema_version", sa.String, nullable=False, server_default="1"),
        sa.Column("prompt_version", sa.String, nullable=False, server_default=""),
        sa.Column("agent", sa.String, nullable=False),   # "amin" | "leila"
        # Decision fields.
        sa.Column("action", sa.String, nullable=False),
        sa.Column("department", sa.String, nullable=True),
        sa.Column("language", sa.String, nullable=True),
        sa.Column("urgency", sa.String, nullable=True),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("rule_hit", sa.String, nullable=True),
        sa.Column("reason", sa.String, nullable=True),
        sa.Column("decision_hash", sa.String, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True), nullable=False,
        ),
    )
    op.create_index("ix_decisions_id", "decisions", ["id"])
    op.create_index("ix_decisions_tenant_id", "decisions", ["tenant_id"])
    op.create_index("ix_decisions_message_id", "decisions", ["message_id"])
    op.create_index("ix_decisions_decision_hash", "decisions", ["decision_hash"])


def downgrade() -> None:
    op.drop_table("decisions")
