"""Follow-up Phase 7 – buyer features: fifteen new tables

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-08 00:00:00.000000

Creates:
  message_assignments  – B1 shared inbox assignment
  internal_notes       – B2 internal notes (never sent)
  draft_locks          – B3 collision lock
  snoozed_messages     – B4 snooze / B6 after-hours hold
  vip_senders          – B5 VIP sender list
  reply_snippets       – B7 saved reply snippets
  department_sla       – B8 per-department SLA
  delivery_failures    – B10 bounce / failure reasons
  vacation_responders  – B11 vacation responder
  user_roles           – B13 role split
  digest_subscriptions – B15 digest subscription
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing_tables = set(insp.get_table_names())

    if "message_assignments" not in existing_tables:
        op.create_table(
            "message_assignments",
            sa.Column("id", sa.Integer, primary_key=True, index=True),
            sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("message_id", sa.Integer, sa.ForeignKey("messages.id"), nullable=False, index=True),
            sa.Column("assigned_to", sa.String, nullable=False),
            sa.Column("assigned_by", sa.String, nullable=True),
            sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("tenant_id", "message_id", name="uq_assignment_tenant_message"),
        )

    if "internal_notes" not in existing_tables:
        op.create_table(
            "internal_notes",
            sa.Column("id", sa.Integer, primary_key=True, index=True),
            sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("message_id", sa.Integer, sa.ForeignKey("messages.id"), nullable=False, index=True),
            sa.Column("author", sa.String, nullable=False),
            sa.Column("body", sa.Text, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )

    if "draft_locks" not in existing_tables:
        op.create_table(
            "draft_locks",
            sa.Column("id", sa.Integer, primary_key=True, index=True),
            sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("draft_id", sa.Integer, sa.ForeignKey("drafts.id"), nullable=False, index=True),
            sa.Column("locked_by", sa.String, nullable=False),
            sa.Column("locked_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("tenant_id", "draft_id", name="uq_lock_tenant_draft"),
        )

    if "snoozed_messages" not in existing_tables:
        op.create_table(
            "snoozed_messages",
            sa.Column("id", sa.Integer, primary_key=True, index=True),
            sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("message_id", sa.Integer, sa.ForeignKey("messages.id"), nullable=False, index=True),
            sa.Column("wake_at", sa.DateTime(timezone=True), nullable=False, index=True),
            sa.Column("snoozed_by", sa.String, nullable=True),
            sa.Column("reason", sa.String, nullable=False, server_default="snooze"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )

    if "vip_senders" not in existing_tables:
        op.create_table(
            "vip_senders",
            sa.Column("id", sa.Integer, primary_key=True, index=True),
            sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("pattern", sa.String, nullable=False),
            sa.Column("label", sa.String, nullable=True),
            sa.Column("added_by", sa.String, nullable=True),
            sa.Column("added_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("tenant_id", "pattern", name="uq_vip_tenant_pattern"),
        )

    if "reply_snippets" not in existing_tables:
        op.create_table(
            "reply_snippets",
            sa.Column("id", sa.Integer, primary_key=True, index=True),
            sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("title", sa.String, nullable=False),
            sa.Column("body", sa.Text, nullable=False),
            sa.Column("created_by", sa.String, nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )

    if "department_sla" not in existing_tables:
        op.create_table(
            "department_sla",
            sa.Column("id", sa.Integer, primary_key=True, index=True),
            sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("department", sa.String, nullable=False),
            sa.Column("critical_hours", sa.Integer, nullable=False, server_default="1"),
            sa.Column("high_hours", sa.Integer, nullable=False, server_default="4"),
            sa.Column("medium_hours", sa.Integer, nullable=False, server_default="24"),
            sa.Column("low_hours", sa.Integer, nullable=False, server_default="72"),
            sa.UniqueConstraint("tenant_id", "department", name="uq_dept_sla_tenant_dept"),
        )

    if "delivery_failures" not in existing_tables:
        op.create_table(
            "delivery_failures",
            sa.Column("id", sa.Integer, primary_key=True, index=True),
            sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("message_id", sa.Integer, sa.ForeignKey("messages.id"), nullable=True, index=True),
            sa.Column("draft_id", sa.Integer, sa.ForeignKey("drafts.id"), nullable=True, index=True),
            sa.Column("recipient", sa.String, nullable=True),
            sa.Column("smtp_code", sa.Integer, nullable=True),
            sa.Column("reason", sa.Text, nullable=True),
            sa.Column("failed_at", sa.DateTime(timezone=True), nullable=False),
        )

    if "vacation_responders" not in existing_tables:
        op.create_table(
            "vacation_responders",
            sa.Column("id", sa.Integer, primary_key=True, index=True),
            sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id"), nullable=False, unique=True, index=True),
            sa.Column("active", sa.Boolean, nullable=False, server_default="0"),
            sa.Column("template_id", sa.String, nullable=False),
            sa.Column("start_date", sa.DateTime(timezone=True), nullable=True),
            sa.Column("end_date", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_by", sa.String, nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )

    if "user_roles" not in existing_tables:
        op.create_table(
            "user_roles",
            sa.Column("id", sa.Integer, primary_key=True, index=True),
            sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False, index=True),
            sa.Column("role", sa.String, nullable=False, server_default="operator"),
            sa.Column("granted_by", sa.String, nullable=True),
            sa.Column("granted_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("tenant_id", "user_id", name="uq_role_tenant_user"),
        )

    if "digest_subscriptions" not in existing_tables:
        op.create_table(
            "digest_subscriptions",
            sa.Column("id", sa.Integer, primary_key=True, index=True),
            sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False, index=True),
            sa.Column("recipient_email", sa.String, nullable=False),
            sa.Column("send_mode", sa.String, nullable=False, server_default="shadow"),
            sa.Column("active", sa.Boolean, nullable=False, server_default="1"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("tenant_id", "user_id", name="uq_digest_tenant_user"),
        )


def downgrade() -> None:
    for table in [
        "digest_subscriptions",
        "user_roles",
        "vacation_responders",
        "delivery_failures",
        "department_sla",
        "reply_snippets",
        "vip_senders",
        "snoozed_messages",
        "draft_locks",
        "internal_notes",
        "message_assignments",
    ]:
        bind = op.get_bind()
        insp = sa.inspect(bind)
        if table in insp.get_table_names():
            op.drop_table(table)
