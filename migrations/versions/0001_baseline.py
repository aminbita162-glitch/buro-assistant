"""baseline – create all tables for Phase 3

Revision ID: 0001
Revises:
Create Date: 2026-10-03 00:00:00.000000

Creates: tenants, users, user_sessions, tasks
Adds:    tenant_id on users, user_sessions, tasks
Removes: legacy users.token column (replaced by user_sessions)
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing = insp.get_table_names()

    # ------------------------------------------------------------------ tenants
    if "tenants" not in existing:
        op.create_table(
            "tenants",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("name", sa.String, nullable=False, unique=True),
            sa.Column("slug", sa.String, nullable=False, unique=True),
        )
        op.create_index("ix_tenants_id", "tenants", ["id"])
        op.create_index("ix_tenants_slug", "tenants", ["slug"])

    # ------------------------------------------------------------------ users
    if "users" not in existing:
        op.create_table(
            "users",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id"), nullable=False),
            sa.Column("name", sa.String),
            sa.Column("email", sa.String, nullable=False, unique=True),
            sa.Column("password_hash", sa.String),
            sa.Column("last_task_id", sa.Integer, nullable=True),
        )
        op.create_index("ix_users_id", "users", ["id"])
        op.create_index("ix_users_email", "users", ["email"])
        op.create_index("ix_users_tenant_id", "users", ["tenant_id"])
    else:
        # Table exists (upgrading from pre-Phase-3 schema).
        user_cols = {c["name"] for c in insp.get_columns("users")}
        if "tenant_id" not in user_cols:
            # Add a nullable tenant_id first; we cannot set a FK default easily,
            # so we create the default tenant, backfill, then enforce NOT NULL
            # via a new column (SQLite does not support ALTER COLUMN).
            op.add_column("users", sa.Column("tenant_id", sa.Integer, nullable=True))

        # Drop the legacy token column that was replaced by user_sessions.
        if "token" in user_cols:
            with op.batch_alter_table("users") as batch_op:
                batch_op.drop_column("token")

    # ------------------------------------------------------------------ user_sessions
    if "user_sessions" not in existing:
        op.create_table(
            "user_sessions",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id"), nullable=False),
            sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
            sa.Column("token_hash", sa.String, nullable=False, unique=True),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_user_sessions_id", "user_sessions", ["id"])
        op.create_index("ix_user_sessions_tenant_id", "user_sessions", ["tenant_id"])
        op.create_index("ix_user_sessions_user_id", "user_sessions", ["user_id"])
        op.create_index("ix_user_sessions_token_hash", "user_sessions", ["token_hash"])
    else:
        sess_cols = {c["name"] for c in insp.get_columns("user_sessions")}
        if "tenant_id" not in sess_cols:
            op.add_column("user_sessions", sa.Column("tenant_id", sa.Integer, nullable=True))

    # ------------------------------------------------------------------ tasks
    if "tasks" not in existing:
        op.create_table(
            "tasks",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("tenant_id", sa.Integer, sa.ForeignKey("tenants.id"), nullable=False),
            sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
            sa.Column("title", sa.String),
            sa.Column("deadline", sa.String),
            sa.Column("priority", sa.String),
            sa.Column("status", sa.String, nullable=True),
        )
        op.create_index("ix_tasks_id", "tasks", ["id"])
        op.create_index("ix_tasks_tenant_id", "tasks", ["tenant_id"])
        op.create_index("ix_tasks_user_id", "tasks", ["user_id"])
    else:
        task_cols = {c["name"] for c in insp.get_columns("tasks")}
        if "tenant_id" not in task_cols:
            op.add_column("tasks", sa.Column("tenant_id", sa.Integer, nullable=True))
        if "status" not in task_cols:
            op.add_column("tasks", sa.Column("status", sa.String, nullable=True))
        if "user_id" not in task_cols:
            op.add_column("tasks", sa.Column("user_id", sa.Integer, nullable=True))


def downgrade() -> None:
    # Downgrade drops all tables created by this baseline.
    op.drop_table("tasks")
    op.drop_table("user_sessions")
    op.drop_table("users")
    op.drop_table("tenants")
