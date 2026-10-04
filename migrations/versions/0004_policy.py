"""Phase 6 – policy: add drafts, approval_queue, audit_log tables

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-03 00:00:00.000000

Rows closed:
  16 – template registry (no schema required — registry is in-memory)
  17 – auto-reply off by default (policy config, no schema required)
  18 – receipt template (in-memory, no schema required)
  19 – business-hours calendar (policy config, no schema required)
  20 – SLA clock (policy config, no schema required)
  21 – human approval queue  → approval_queue table
  22 – append-only audit log → audit_log table
  48 – shadow mode           → drafts table (state = "shadow" | "draft")
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing = insp.get_table_names()

    # ------------------------------------------------------------------
    # drafts (row 48 – shadow mode)
    # ------------------------------------------------------------------
    if "drafts" not in existing:
        op.create_table(
            "drafts",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column(
                "tenant_id", sa.Integer,
                sa.ForeignKey("tenants.id"), nullable=False,
            ),
            sa.Column(
                "message_id", sa.Integer,
                sa.ForeignKey("messages.id"), nullable=True,
            ),
            sa.Column("subject", sa.String, nullable=False, server_default=""),
            sa.Column("body", sa.Text, nullable=False, server_default=""),
            sa.Column("template_id", sa.String, nullable=True),
            sa.Column("language", sa.String, nullable=True),
            sa.Column("decision_hash", sa.String, nullable=True),
            # state: draft | shadow | sent | approved | rejected
            sa.Column("state", sa.String, nullable=False, server_default="draft"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_drafts_id", "drafts", ["id"])
        op.create_index("ix_drafts_tenant_id", "drafts", ["tenant_id"])
        op.create_index("ix_drafts_message_id", "drafts", ["message_id"])

    # ------------------------------------------------------------------
    # approval_queue (row 21 – human approval queue)
    # ------------------------------------------------------------------
    if "approval_queue" not in existing:
        op.create_table(
            "approval_queue",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column(
                "tenant_id", sa.Integer,
                sa.ForeignKey("tenants.id"), nullable=False,
            ),
            sa.Column(
                "message_id", sa.Integer,
                sa.ForeignKey("messages.id"), nullable=True,
            ),
            sa.Column("subject", sa.String, nullable=False, server_default=""),
            sa.Column("body", sa.Text, nullable=False, server_default=""),
            sa.Column("template_id", sa.String, nullable=True),
            # state: pending | approved | rejected
            sa.Column("state", sa.String, nullable=False, server_default="pending"),
            sa.Column("reason", sa.String, nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("resolved_by", sa.String, nullable=True),
        )
        op.create_index("ix_approval_queue_id", "approval_queue", ["id"])
        op.create_index("ix_approval_queue_tenant_id", "approval_queue", ["tenant_id"])
        op.create_index("ix_approval_queue_message_id", "approval_queue", ["message_id"])

    # ------------------------------------------------------------------
    # audit_log (row 22 – append-only audit log)
    # ------------------------------------------------------------------
    if "audit_log" not in existing:
        op.create_table(
            "audit_log",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column(
                "tenant_id", sa.Integer,
                sa.ForeignKey("tenants.id"), nullable=False,
            ),
            sa.Column("event", sa.String, nullable=False),
            sa.Column(
                "message_id", sa.Integer,
                sa.ForeignKey("messages.id"), nullable=True,
            ),
            sa.Column("actor", sa.String, nullable=True),
            sa.Column("detail", sa.Text, nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_audit_log_id", "audit_log", ["id"])
        op.create_index("ix_audit_log_tenant_id", "audit_log", ["tenant_id"])
        op.create_index("ix_audit_log_event", "audit_log", ["event"])
        op.create_index("ix_audit_log_created_at", "audit_log", ["created_at"])
        op.create_index("ix_audit_log_message_id", "audit_log", ["message_id"])


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("approval_queue")
    op.drop_table("drafts")
