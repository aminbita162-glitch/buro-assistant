"""
Phase 2 security tests.

Covered checklist rows:
  31 – live (/health) and ready (/ready) health checks
  33 – Argon2id password hashes; legacy SHA-256 one-time upgrade
  34 – session table, hashed token, expiry, rotation on login
  35 – rate limits on signup, login, and model routes
  36 – security headers and CSP on every response
  37 – CORS allowlist from the environment
  38 – message and task fields rendered as text, not HTML

These tests run without a real database or OpenAI key.
See tests/conftest.py for the SQLite + cross-thread patch applied before import.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import (  # noqa: E402
    app,
    Base,
    engine,
    SessionLocal,
    Tenant,
    User,
    UserSession,
    hash_password,
    verify_password,
    _sha256_hex,
    _token_hash,
    create_session,
    get_session_user,
    SESSION_TTL_HOURS,
    limiter,
)
from argon2.exceptions import VerifyMismatchError  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_db():
    """Re-create all tables before each test and clear the rate limiter."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    # Reset in-memory rate limit counters between tests
    try:
        limiter._storage.reset()
    except Exception:
        pass
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _ensure_tenant(slug="test", name="Test Tenant") -> int:
    """Return the id of a tenant, creating it if absent."""
    db = SessionLocal()
    try:
        t = db.query(Tenant).filter(Tenant.slug == slug).first()
        if not t:
            t = Tenant(name=name, slug=slug)
            db.add(t)
            db.commit()
            db.refresh(t)
        return t.id
    finally:
        db.close()


def _make_user(name="Alice", email="alice@example.com", password="secret99",
               tenant_id=None):
    """Insert a user with an Argon2id hash and return (user, raw_password)."""
    if tenant_id is None:
        tenant_id = _ensure_tenant()
    db = SessionLocal()
    try:
        u = User(tenant_id=tenant_id, name=name, email=email,
                 password_hash=hash_password(password))
        db.add(u)
        db.commit()
        db.refresh(u)
        return u, password
    finally:
        db.close()


def _make_legacy_user(name="Bob", email="bob@example.com", password="legacy123"):
    """Insert a user with a legacy SHA-256 hash."""
    tenant_id = _ensure_tenant()
    db = SessionLocal()
    try:
        u = User(tenant_id=tenant_id, name=name, email=email,
                 password_hash=_sha256_hex(password))
        db.add(u)
        db.commit()
        db.refresh(u)
        return u, password
    finally:
        db.close()


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _login(client, email, password):
    """Helper: POST /auth/login and return the raw token."""
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["token"]


# ---------------------------------------------------------------------------
# Row 31 – live and ready health checks
# ---------------------------------------------------------------------------

class TestHealthChecks:
    def test_liveness_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_readiness_ok(self, client):
        """SQLite is always available during tests."""
        r = client.get("/ready")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_liveness_no_auth_required(self, client):
        """Health endpoints must not require authentication."""
        r = client.get("/health")
        assert r.status_code == 200

    def test_readiness_no_auth_required(self, client):
        r = client.get("/ready")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Row 33 – Argon2id hashes; legacy SHA-256 one-time upgrade
# ---------------------------------------------------------------------------

class TestPasswordHashing:
    def test_hash_is_argon2id(self):
        h = hash_password("my-password")
        assert h.startswith("$argon2id"), f"Expected Argon2id prefix, got: {h[:20]}"

    def test_correct_argon2id_verifies(self):
        h = hash_password("correct")
        assert verify_password(h, "correct") is True

    def test_wrong_argon2id_raises(self):
        h = hash_password("correct")
        with pytest.raises(VerifyMismatchError):
            verify_password(h, "wrong")

    def test_correct_legacy_sha256_verifies(self):
        h = _sha256_hex("legacy-pass")
        assert verify_password(h, "legacy-pass") is True

    def test_wrong_legacy_sha256_raises(self):
        h = _sha256_hex("legacy-pass")
        with pytest.raises(VerifyMismatchError):
            verify_password(h, "bad-pass")

    def test_signup_stores_argon2id(self, client):
        r = client.post("/auth/signup",
                        json={"name": "Tester", "email": "t@x.com", "password": "pass99"})
        assert r.status_code == 200, r.text
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.email == "t@x.com").first()
            assert u is not None
            assert u.password_hash.startswith("$argon2id")
        finally:
            db.close()

    def test_legacy_hash_upgraded_on_login(self, client):
        _user, password = _make_legacy_user()
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.email == "bob@example.com").first()
            assert not u.password_hash.startswith("$argon2")
        finally:
            db.close()

        # Login triggers upgrade
        r = client.post("/auth/login",
                        json={"email": "bob@example.com", "password": password})
        assert r.status_code == 200, r.text

        db = SessionLocal()
        try:
            u = db.query(User).filter(User.email == "bob@example.com").first()
            assert u.password_hash.startswith("$argon2id"), "Hash not upgraded"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Row 34 – session table, hashed token, expiry, rotation on login
# ---------------------------------------------------------------------------

class TestSessions:
    def test_login_creates_session_row(self, client):
        _make_user()
        token = _login(client, "alice@example.com", "secret99")
        db = SessionLocal()
        try:
            count = db.query(UserSession).count()
            assert count == 1
            session = db.query(UserSession).first()
            # Token is stored hashed, not raw
            assert session.token_hash == _token_hash(token)
            assert session.token_hash != token
        finally:
            db.close()

    def test_token_stored_as_hash_not_plaintext(self, client):
        _make_user()
        token = _login(client, "alice@example.com", "secret99")
        db = SessionLocal()
        try:
            session = db.query(UserSession).first()
            assert session.token_hash != token
            assert len(session.token_hash) == 64  # SHA-256 hex
        finally:
            db.close()

    def test_session_has_expiry(self, client):
        _make_user()
        _login(client, "alice@example.com", "secret99")
        db = SessionLocal()
        try:
            session = db.query(UserSession).first()
            assert session.expires_at is not None
            expected_max = datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS + 1)
            expires = session.expires_at.replace(tzinfo=timezone.utc)
            assert expires > datetime.now(timezone.utc)
            assert expires <= expected_max
        finally:
            db.close()

    def test_rotation_on_second_login(self, client):
        _make_user()
        token1 = _login(client, "alice@example.com", "secret99")
        token2 = _login(client, "alice@example.com", "secret99")
        assert token1 != token2
        db = SessionLocal()
        try:
            count = db.query(UserSession).count()
            assert count == 1  # old session deleted
            session = db.query(UserSession).first()
            assert session.token_hash == _token_hash(token2)
        finally:
            db.close()

    def test_expired_session_rejected(self):
        tenant_id = _ensure_tenant()
        db = SessionLocal()
        try:
            u = User(tenant_id=tenant_id, name="Eve", email="eve@example.com",
                     password_hash=hash_password("pw"))
            db.add(u)
            db.commit()
            db.refresh(u)
            # Insert an already-expired session
            raw = "expiredtoken123"
            s = UserSession(
                tenant_id=tenant_id,
                user_id=u.id,
                token_hash=_token_hash(raw),
                expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            )
            db.add(s)
            db.commit()
            result = get_session_user(db, raw)
            assert result is None
        finally:
            db.close()

    def test_auth_me_requires_valid_session(self, client):
        r = client.get("/auth/me", headers={"Authorization": "Bearer faketoken"})
        assert r.status_code == 401

    def test_auth_me_returns_user_with_valid_session(self, client):
        _make_user()
        token = _login(client, "alice@example.com", "secret99")
        r = client.get("/auth/me", headers=_auth_header(token))
        assert r.status_code == 200
        assert r.json()["user"]["email"] == "alice@example.com"

    def test_logout_deletes_session(self, client):
        _make_user()
        token = _login(client, "alice@example.com", "secret99")
        client.post("/auth/logout", headers=_auth_header(token))
        db = SessionLocal()
        try:
            assert db.query(UserSession).count() == 0
        finally:
            db.close()
        # Token no longer works
        r = client.get("/auth/me", headers=_auth_header(token))
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# Row 35 – rate limits on signup, login, and model routes
# Each class gets a fresh client so rate counters from previous classes
# do not bleed in (reset_db fixture clears limiter storage).
# ---------------------------------------------------------------------------

class TestRateLimits:
    def test_signup_rate_limited(self, client):
        """Signup is limited to 5/minute; the 6th request returns 429."""
        for i in range(5):
            client.post("/auth/signup",
                        json={"name": f"U{i}", "email": f"u{i}@x.com",
                              "password": "pass99"})
        r = client.post("/auth/signup",
                        json={"name": "X", "email": "x@x.com", "password": "pass99"})
        assert r.status_code == 429

    def test_login_rate_limited(self, client):
        """Login is limited to 10/minute; the 11th request returns 429."""
        for _ in range(10):
            client.post("/auth/login",
                        json={"email": "nobody@x.com", "password": "wrong"})
        r = client.post("/auth/login",
                        json={"email": "nobody@x.com", "password": "wrong"})
        assert r.status_code == 429

    def test_assistant_rate_limited(self, client):
        """Assistant model route is limited to 30/minute."""
        _make_user()
        token = _login(client, "alice@example.com", "secret99")
        headers = _auth_header(token)
        # Exhaust the 30-request limit (list tasks hits no model calls, passes fast)
        for _ in range(30):
            client.post("/assistant", json={"text": "show my tasks"}, headers=headers)
        r = client.post("/assistant", json={"text": "show my tasks"}, headers=headers)
        assert r.status_code == 429

    def test_analyze_rate_limited(self, client):
        """Analyze model route is limited to 20/minute."""
        _make_user()
        token = _login(client, "alice@example.com", "secret99")
        headers = _auth_header(token)
        for _ in range(20):
            client.post("/analyze", json={"text": "email body"}, headers=headers)
        r = client.post("/analyze", json={"text": "email body"}, headers=headers)
        assert r.status_code == 429


# ---------------------------------------------------------------------------
# Row 36 – security headers and CSP
# ---------------------------------------------------------------------------

class TestSecurityHeaders:
    def _check_headers(self, response):
        h = response.headers
        assert h.get("X-Content-Type-Options") == "nosniff"
        assert h.get("X-Frame-Options") == "DENY"
        assert "Content-Security-Policy" in h
        csp = h["Content-Security-Policy"]
        assert "default-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp
        assert h.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    def test_headers_on_health(self, client):
        self._check_headers(client.get("/health"))

    def test_headers_on_ready(self, client):
        self._check_headers(client.get("/ready"))

    def test_headers_on_root(self, client):
        self._check_headers(client.get("/"))

    def test_headers_on_api_endpoint(self, client):
        self._check_headers(client.get("/tasks"))

    def test_x_frame_options_deny(self, client):
        r = client.get("/health")
        assert r.headers.get("X-Frame-Options") == "DENY"

    def test_csp_no_frame_ancestors(self, client):
        r = client.get("/health")
        assert "frame-ancestors 'none'" in r.headers.get("Content-Security-Policy", "")


# ---------------------------------------------------------------------------
# Row 37 – CORS allowlist from the environment
# ---------------------------------------------------------------------------

class TestCORS:
    def test_allowed_origin_gets_cors_header(self, client):
        r = client.get("/health",
                       headers={"Origin": "http://localhost:3000"})
        assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_disallowed_origin_no_cors_header(self, client):
        r = client.get("/health",
                       headers={"Origin": "http://evil.example.com"})
        assert r.headers.get("access-control-allow-origin") != "http://evil.example.com"

    def test_preflight_allowed_origin(self, client):
        r = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


# ---------------------------------------------------------------------------
# Row 38 – message and task fields rendered as text, not HTML
# ---------------------------------------------------------------------------

class TestTextRendering:
    def test_serialize_task_returns_strings(self):
        from app.main import serialize_task, Task
        t = Task(
            id=1,
            title="<b>bold</b>",
            deadline="2026-12-01",
            priority="high",
            status="active",
            user_id=1,
        )
        result = serialize_task(t)
        # Value must be the raw string — not HTML-escaped by the server
        assert result["title"] == "<b>bold</b>"
        assert isinstance(result["title"], str)
        assert isinstance(result["deadline"], str)
        assert isinstance(result["priority"], str)

    def test_index_html_uses_text_not_innerhtml(self):
        """
        Verify the task list renderer in index.html does not use innerHTML
        with task field data.  The fix replaced the interpolated innerHTML
        template literal with individual DOM text node assignments.
        """
        with open("index.html", encoding="utf-8") as f:
            source = f.read()

        # The old vulnerable pattern interpolated task.title directly into innerHTML
        assert "task.title}<br>" not in source, \
            "index.html still uses innerHTML with task.title"

        # The safe pattern must be present
        assert "textContent" in source or "createTextNode" in source, \
            "index.html does not use textContent or createTextNode for task fields"
