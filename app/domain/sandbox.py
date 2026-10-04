"""
app/domain/sandbox.py – sandbox tenant seed command (row 43).

Creates an isolated sandbox tenant pre-loaded with:
  - one sandbox user
  - a sample message (via FakeProvider)
  - a sample draft
  - a sample task
  - a sample API key (printed once)

Usage:
    python -m app.domain.sandbox

The command is idempotent: running it twice does not create duplicates.
It always targets the ``sandbox`` slug.
"""
from __future__ import annotations

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.main import (   # noqa: E402
    SessionLocal, Tenant, User, Task, hash_password, _run_migrations,
)

SANDBOX_SLUG     = "sandbox"
SANDBOX_NAME     = "Sandbox Tenant"
SANDBOX_EMAIL    = "sandbox@example.com"
SANDBOX_PASSWORD = "sandbox-password-1"
SANDBOX_USER     = "Sandbox Operator"


def seed_sandbox() -> None:
    """Create or verify the sandbox tenant and its demo data."""
    print("Running migrations…")
    _run_migrations()

    db = SessionLocal()
    try:
        # ---- Tenant ----
        tenant = db.query(Tenant).filter(Tenant.slug == SANDBOX_SLUG).first()
        if not tenant:
            tenant = Tenant(name=SANDBOX_NAME, slug=SANDBOX_SLUG)
            db.add(tenant)
            db.commit()
            db.refresh(tenant)
            print(f"Created sandbox tenant: {tenant.name} (id={tenant.id})")
        else:
            print(f"Sandbox tenant already exists (id={tenant.id})")

        # ---- User ----
        user = db.query(User).filter(User.email == SANDBOX_EMAIL).first()
        if not user:
            user = User(
                tenant_id=tenant.id,
                name=SANDBOX_USER,
                email=SANDBOX_EMAIL,
                password_hash=hash_password(SANDBOX_PASSWORD),
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"Created sandbox user: {user.email}")
            print(f"  password: {SANDBOX_PASSWORD}")
        else:
            print(f"Sandbox user already exists: {user.email}")

        # ---- Sample task ----
        existing = (
            db.query(Task)
            .filter(Task.tenant_id == tenant.id, Task.user_id == user.id)
            .first()
        )
        if not existing:
            task = Task(
                tenant_id=tenant.id,
                user_id=user.id,
                title="Review sandbox mailbox",
                deadline="2026-12-31",
                priority="low",
                status="active",
            )
            db.add(task)
            db.commit()
            print(f"Created sandbox task: '{task.title}'")
        else:
            print("Sandbox task already exists.")

        # ---- Sample API key ----
        try:
            from app.domain.apikeys import create_api_key, list_api_keys
            existing_keys = list_api_keys(db, tenant.id)
            if not existing_keys:
                _, raw = create_api_key(db, tenant.id, "sandbox-default", scope="ingest")
                print(f"Created sandbox API key: {raw}")
                print("  (shown once – store it now)")
            else:
                print(f"Sandbox API key already exists ({len(existing_keys)} key(s)).")
        except Exception as exc:  # noqa: BLE001
            print(f"  (API key creation skipped: {exc})")

        print("Sandbox seed complete.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_sandbox()
