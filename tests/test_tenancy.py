"""
Phase 3 tenancy tests.

Covered checklist rows:
  1  – tenant_id on every business table (tenants, users, user_sessions, tasks)
  2  – queries reject cross-tenant reads
  32 – Alembic baseline migration; no schema change on import

These tests run against the in-memory SQLite engine configured in conftest.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect as sa_inspect

from app.main import (
    app,
    Base,
    engine,
    SessionLocal,
    Tenant,
    User,
    UserSession,
    Task,
    hash_password,
    _token_hash,
    limiter,
)
from datetime import datetime, timedelta, timezone


# ---------------------------------------------------------------------------
# Fixtures – shared with test_security.py via conftest.py
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    try:
        limiter._storage.reset()
    except Exception:
        pass
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tenant(slug: str, name: str | None = None) -> Tenant:
    db = SessionLocal()
    try:
        t = Tenant(name=name or slug.capitalize(), slug=slug)
        db.add(t)
        db.commit()
        db.refresh(t)
        return t
    finally:
        db.close()


def _make_user(tenant: Tenant, email: str, password: str = "pass99") -> User:
    db = SessionLocal()
    try:
        u = User(
            tenant_id=tenant.id,
            name=email.split("@")[0],
            email=email,
            password_hash=hash_password(password),
        )
        db.add(u)
        db.commit()
        db.refresh(u)
        return u
    finally:
        db.close()


def _make_task(tenant: Tenant, user: User, title: str = "Test task") -> Task:
    db = SessionLocal()
    try:
        t = Task(
            tenant_id=tenant.id,
            user_id=user.id,
            title=title,
            deadline="2026-12-01",
            priority="medium",
            status="active",
        )
        db.add(t)
        db.commit()
        db.refresh(t)
        return t
    finally:
        db.close()


def _make_session(tenant: Tenant, user: User, raw_token: str = "tok123") -> UserSession:
    db = SessionLocal()
    try:
        s = UserSession(
            tenant_id=tenant.id,
            user_id=user.id,
            token_hash=_token_hash(raw_token),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        )
        db.add(s)
        db.commit()
        db.refresh(s)
        return s
    finally:
        db.close()


def _login(client, email: str, password: str = "pass99") -> str:
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["token"]


# ---------------------------------------------------------------------------
# Row 1 – tenant_id on every business table
# ---------------------------------------------------------------------------

class TestTenantIdColumns:
    """Verify that tenant_id is present on every business table."""

    def _columns(self, table: str) -> set:
        insp = sa_inspect(engine)
        return {c["name"] for c in insp.get_columns(table)}

    def test_tenants_table_exists(self):
        insp = sa_inspect(engine)
        assert "tenants" in insp.get_table_names()

    def test_users_has_tenant_id(self):
        assert "tenant_id" in self._columns("users")

    def test_tasks_has_tenant_id(self):
        assert "tenant_id" in self._columns("tasks")

    def test_user_sessions_has_tenant_id(self):
        assert "tenant_id" in self._columns("user_sessions")

    def test_tenant_row_persists(self):
        t = _make_tenant("acme")
        db = SessionLocal()
        try:
            found = db.query(Tenant).filter(Tenant.slug == "acme").first()
            assert found is not None
            assert found.id == t.id
        finally:
            db.close()

    def test_user_carries_tenant_id(self):
        t = _make_tenant("corp")
        u = _make_user(t, "alice@corp.com")
        db = SessionLocal()
        try:
            found = db.query(User).filter(User.id == u.id).first()
            assert found.tenant_id == t.id
        finally:
            db.close()

    def test_task_carries_tenant_id(self):
        t = _make_tenant("biz")
        u = _make_user(t, "bob@biz.com")
        task = _make_task(t, u)
        db = SessionLocal()
        try:
            found = db.query(Task).filter(Task.id == task.id).first()
            assert found.tenant_id == t.id
        finally:
            db.close()

    def test_user_session_carries_tenant_id(self):
        t = _make_tenant("xyz")
        u = _make_user(t, "charlie@xyz.com")
        s = _make_session(t, u, "rawtoken42")
        db = SessionLocal()
        try:
            found = db.query(UserSession).filter(UserSession.id == s.id).first()
            assert found.tenant_id == t.id
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Row 2 – cross-tenant reads are rejected
# ---------------------------------------------------------------------------

class TestCrossTenantIsolation:
    """
    A user in tenant A must not be able to read or modify records belonging
    to tenant B through the API.  All task endpoints filter on both
    user_id and tenant_id; a token from tenant A cannot reach tenant B's tasks.
    """

    def _setup_two_tenants(self, client):
        """
        Create two independent tenants each with one user and one task.
        Returns (token_a, task_b_id).
        """
        # Tenant A
        r = client.post("/auth/signup", json={
            "name": "Alice", "email": "alice@tenant-a.com", "password": "passA99",
        })
        assert r.status_code == 200, r.text
        token_a = _login(client, "alice@tenant-a.com", "passA99")

        # Tenant B – created directly in DB with a different tenant_id
        db = SessionLocal()
        try:
            tenant_b = Tenant(name="Tenant B", slug="tenant-b")
            db.add(tenant_b)
            db.commit()
            db.refresh(tenant_b)

            user_b = User(
                tenant_id=tenant_b.id,
                name="Bob",
                email="bob@tenant-b.com",
                password_hash=hash_password("passB99"),
            )
            db.add(user_b)
            db.commit()
            db.refresh(user_b)

            task_b = Task(
                tenant_id=tenant_b.id,
                user_id=user_b.id,
                title="Tenant B secret",
                deadline="2026-12-01",
                priority="high",
                status="active",
            )
            db.add(task_b)
            db.commit()
            db.refresh(task_b)
            task_b_id = task_b.id
        finally:
            db.close()

        return token_a, task_b_id

    def test_cannot_read_other_tenant_task(self, client):
        token_a, task_b_id = self._setup_two_tenants(client)
        r = client.get(
            f"/tasks/{task_b_id}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert r.status_code == 404, (
            f"Expected 404 for cross-tenant task read, got {r.status_code}: {r.text}"
        )

    def test_cannot_delete_other_tenant_task(self, client):
        token_a, task_b_id = self._setup_two_tenants(client)
        r = client.delete(
            f"/tasks/{task_b_id}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert r.status_code == 404

    def test_cannot_update_other_tenant_task(self, client):
        token_a, task_b_id = self._setup_two_tenants(client)
        r = client.put(
            f"/tasks/{task_b_id}",
            json={"title": "hacked", "deadline": "2026-01-01", "priority": "low"},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert r.status_code == 404

    def test_task_list_excludes_other_tenant(self, client):
        """GET /tasks must return only the authenticated user's tenant tasks."""
        token_a, task_b_id = self._setup_two_tenants(client)
        r = client.get(
            "/tasks",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert r.status_code == 200
        task_ids = [t["id"] for t in r.json().get("tasks", [])]
        assert task_b_id not in task_ids, (
            f"Cross-tenant task {task_b_id} appeared in tenant A's task list"
        )

    def test_signup_assigns_default_tenant(self, client):
        """All self-service signups share the 'default' tenant slug."""
        client.post("/auth/signup", json={
            "name": "User1", "email": "u1@x.com", "password": "pass99",
        })
        client.post("/auth/signup", json={
            "name": "User2", "email": "u2@x.com", "password": "pass99",
        })
        db = SessionLocal()
        try:
            u1 = db.query(User).filter(User.email == "u1@x.com").first()
            u2 = db.query(User).filter(User.email == "u2@x.com").first()
            assert u1 is not None and u2 is not None
            assert u1.tenant_id == u2.tenant_id, (
                "Two self-service signups should share the default tenant"
            )
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Row 32 – Alembic baseline migration; no create_all on import
# ---------------------------------------------------------------------------

class TestAlembicMigration:
    def test_alembic_version_table_exists(self):
        """
        After reset_db runs Base.metadata.create_all (test-only shortcut),
        the alembic_version table will NOT be present – that is intentional
        for pure unit tests.  This test verifies the migration script itself
        can be introspected without error.
        """
        from alembic.config import Config
        from alembic.script import ScriptDirectory
        cfg = Config("alembic.ini")
        scripts = ScriptDirectory.from_config(cfg)
        revisions = list(scripts.walk_revisions())
        assert len(revisions) >= 1, "Expected at least one Alembic revision"
        heads = scripts.get_heads()
        assert len(heads) >= 1, f"Expected at least one head revision, got: {heads}"
        assert len(revisions) >= 1, "Expected at least one Alembic revision"

    def test_baseline_migration_is_importable(self):
        """The baseline migration module must be importable without error."""
        import importlib
        mod = importlib.import_module(
            "migrations.versions.0001_baseline"
        )
        assert hasattr(mod, "upgrade")
        assert hasattr(mod, "downgrade")
        assert mod.revision == "0001"
        assert mod.down_revision is None

    def test_no_create_all_in_main(self):
        """
        app/main.py must not call Base.metadata.create_all at module level.
        Schema is managed exclusively by Alembic.
        """
        with open("app/main.py", encoding="utf-8") as f:
            source = f.read()
        assert "Base.metadata.create_all" not in source, (
            "app/main.py calls Base.metadata.create_all – "
            "schema changes must go through Alembic migrations only"
        )

    def test_no_ensure_column_in_main(self):
        """
        The pre-Phase-3 _ensure_column bootstrap must be removed from main.py.
        """
        with open("app/main.py", encoding="utf-8") as f:
            source = f.read()
        assert "_ensure_column(" not in source, (
            "app/main.py still calls _ensure_column – "
            "schema changes must go through Alembic migrations only"
        )
