"""
app/domain/seed.py – demo seed command.

Creates a sandbox tenant, a demo user, and a sample task so an operator
can verify the system works end-to-end without real mail credentials.

Usage:
    python -m app.domain.seed

Environment variables required:
    DATABASE_URL   – target database

The command is idempotent: running it twice does not create duplicates.
"""
from __future__ import annotations

import os
import sys

# Ensure the project root is on sys.path when run as a script.
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.main import (  # noqa: E402  – after path setup
    SessionLocal,
    Tenant,
    User,
    Task,
    hash_password,
    _run_migrations,
)

DEMO_TENANT_SLUG = "demo"
DEMO_TENANT_NAME = "Demo Tenant"
DEMO_USER_EMAIL = "demo@example.com"
DEMO_USER_PASSWORD = "demo-password-1"
DEMO_USER_NAME = "Demo Operator"


def seed() -> None:
    """Create or verify the demo tenant, user, and sample task."""
    print("Running migrations…")
    _run_migrations()

    db = SessionLocal()
    try:
        # ---- Tenant ----
        tenant = db.query(Tenant).filter(Tenant.slug == DEMO_TENANT_SLUG).first()
        if not tenant:
            tenant = Tenant(name=DEMO_TENANT_NAME, slug=DEMO_TENANT_SLUG)
            db.add(tenant)
            db.commit()
            db.refresh(tenant)
            print(f"Created tenant: {tenant.name} (id={tenant.id})")
        else:
            print(f"Tenant already exists: {tenant.name} (id={tenant.id})")

        # ---- User ----
        user = db.query(User).filter(User.email == DEMO_USER_EMAIL).first()
        if not user:
            user = User(
                tenant_id=tenant.id,
                name=DEMO_USER_NAME,
                email=DEMO_USER_EMAIL,
                password_hash=hash_password(DEMO_USER_PASSWORD),
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"Created user: {user.email} (id={user.id})")
            print(f"  password: {DEMO_USER_PASSWORD}")
        else:
            print(f"User already exists: {user.email} (id={user.id})")

        # ---- Sample task ----
        existing_task = (
            db.query(Task)
            .filter(Task.tenant_id == tenant.id, Task.user_id == user.id)
            .first()
        )
        if not existing_task:
            task = Task(
                tenant_id=tenant.id,
                user_id=user.id,
                title="Review demo mailbox",
                deadline="2026-10-10",
                priority="medium",
                status="active",
            )
            db.add(task)
            db.commit()
            db.refresh(task)
            print(f"Created sample task: '{task.title}' (id={task.id})")
        else:
            print(f"Sample task already exists (id={existing_task.id})")

        print("Seed complete.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
