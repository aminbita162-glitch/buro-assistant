"""
tests/test_webhook_delivery.py – webhook delivery tests (follow-up Phase 2).

Covers:
  - Signed payload produced correctly (signature header present and valid)
  - Delivery log row written on success
  - Delivery log row written on failure
  - Retry loop: succeeds on second attempt
  - Retry loop: all attempts exhausted → DeliveryResult.success is False
  - dispatch_event delivers to multiple subscriptions
  - dispatch_event skips subscriptions with no secret in the map
  - No secret value appears in DeliveryLog rows
  - delivery_log table exists in schema
"""
from __future__ import annotations

import json
import uuid
import hmac
import hashlib
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from app.main import Base, engine, SessionLocal, Tenant, limiter
from app.domain.webhooks import (
    WebhookSubscription,
    build_signed_payload,
    verify_signature,
    create_subscription,
)
from app.workers.webhook_delivery import (
    DeliveryLog,
    DeliveryResult,
    deliver_event,
    dispatch_event,
    _attempt_post,
    _write_log,
    DEFAULT_MAX_ATTEMPTS,
)


# ---------------------------------------------------------------------------
# Fixtures
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


def _make_tenant(slug: str) -> Tenant:
    db = SessionLocal()
    try:
        t = Tenant(name=slug, slug=slug)
        db.add(t)
        db.commit()
        db.refresh(t)
        return t
    finally:
        db.close()


def _make_subscription(tenant_id: int, url: str = "http://example.invalid/hook") -> tuple:
    """Returns (subscription, plaintext_secret)."""
    db = SessionLocal()
    try:
        sub, secret = create_subscription(db, tenant_id=tenant_id, url=url)
        return sub, secret
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestDeliveryLogTable:
    def test_table_exists(self):
        from sqlalchemy import inspect as sa_inspect
        insp = sa_inspect(engine)
        assert "delivery_log" in insp.get_table_names()

    def test_table_columns(self):
        from sqlalchemy import inspect as sa_inspect
        insp = sa_inspect(engine)
        cols = {c["name"] for c in insp.get_columns("delivery_log")}
        for col in ("id", "tenant_id", "subscription_id", "event_type",
                    "attempt", "status", "http_status", "error_detail",
                    "payload_preview", "delivered_at", "success"):
            assert col in cols, f"missing column: {col}"


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------

class TestSigning:
    def test_build_signed_payload_produces_valid_signature(self):
        secret = "test-secret-abc"
        body, sig = build_signed_payload("message.received", {"id": 1}, secret)
        assert sig.startswith("sha256=")
        assert verify_signature(body, secret, sig)

    def test_wrong_secret_fails_verify(self):
        _, sig = build_signed_payload("evt", {"x": 1}, "secret-a")
        body2, _ = build_signed_payload("evt", {"x": 1}, "secret-b")
        assert not verify_signature(body2, "secret-b", sig)

    def test_payload_contains_event_field(self):
        body, _ = build_signed_payload("message.received", {"msg": "hi"}, "s")
        parsed = json.loads(body)
        assert parsed["event"] == "message.received"
        assert parsed["data"]["msg"] == "hi"


# ---------------------------------------------------------------------------
# _attempt_post (mocked HTTP)
# ---------------------------------------------------------------------------

class TestAttemptPost:
    def test_200_response_returns_success(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            ok, status, err = _attempt_post("http://x.invalid/hook", b'{}', "sha256=abc")
        assert ok is True
        assert status == 200
        assert err is None

    def test_500_response_returns_failure(self):
        import urllib.error
        http_err = urllib.error.HTTPError(
            url="http://x.invalid/hook", code=500, msg="Server Error",
            hdrs=None, fp=None  # type: ignore[arg-type]
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            ok, status, err = _attempt_post("http://x.invalid/hook", b'{}', "sha256=abc")
        assert ok is False
        assert status == 500

    def test_connection_error_returns_failure(self):
        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            ok, status, err = _attempt_post("http://x.invalid/hook", b'{}', "sha256=abc")
        assert ok is False
        assert status is None
        assert "refused" in (err or "")


# ---------------------------------------------------------------------------
# deliver_event
# ---------------------------------------------------------------------------

class TestDeliverEvent:
    def test_success_on_first_attempt(self):
        """Successful delivery: DeliveryResult.success is True, attempts=1."""
        tenant = _make_tenant("wh-ok")
        sub, secret = _make_subscription(tenant.id)
        db = SessionLocal()
        try:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)

            with patch("urllib.request.urlopen", return_value=mock_resp):
                result = deliver_event(
                    db, tenant_id=tenant.id, subscription=sub,
                    event_type="message.received", data={"id": 1},
                    secret=secret, max_attempts=3, base_delay=0,
                )
            assert result.success is True
            assert result.attempts == 1
        finally:
            db.close()

    def test_delivery_log_written_on_success(self):
        tenant = _make_tenant("wh-log-ok")
        sub, secret = _make_subscription(tenant.id)
        db = SessionLocal()
        try:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)

            with patch("urllib.request.urlopen", return_value=mock_resp):
                deliver_event(
                    db, tenant_id=tenant.id, subscription=sub,
                    event_type="draft.approved", data={"draft_id": 7},
                    secret=secret, max_attempts=1, base_delay=0,
                )
            rows = db.query(DeliveryLog).filter(DeliveryLog.tenant_id == tenant.id).all()
            assert len(rows) == 1
            assert rows[0].success is True
            assert rows[0].event_type == "draft.approved"
            assert rows[0].http_status == 200
        finally:
            db.close()

    def test_delivery_log_written_on_failure(self):
        tenant = _make_tenant("wh-log-fail")
        sub, secret = _make_subscription(tenant.id)
        db = SessionLocal()
        try:
            import urllib.error
            http_err = urllib.error.HTTPError(
                url=sub.url, code=503, msg="Service Unavailable",
                hdrs=None, fp=None  # type: ignore[arg-type]
            )
            with patch("urllib.request.urlopen", side_effect=http_err):
                result = deliver_event(
                    db, tenant_id=tenant.id, subscription=sub,
                    event_type="message.received", data={},
                    secret=secret, max_attempts=1, base_delay=0,
                )
            assert result.success is False
            rows = db.query(DeliveryLog).filter(DeliveryLog.tenant_id == tenant.id).all()
            assert len(rows) == 1
            assert rows[0].success is False
            assert rows[0].http_status == 503
        finally:
            db.close()

    def test_retry_succeeds_on_second_attempt(self):
        """First attempt fails, second succeeds: attempts == 2, success True."""
        tenant = _make_tenant("wh-retry")
        sub, secret = _make_subscription(tenant.id)
        db = SessionLocal()
        call_count = [0]
        try:
            import urllib.error

            def _side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise urllib.error.HTTPError(
                        url=sub.url, code=503, msg="Retry me",
                        hdrs=None, fp=None  # type: ignore[arg-type]
                    )
                mock_resp = MagicMock()
                mock_resp.status = 200
                mock_resp.__enter__ = lambda s: s
                mock_resp.__exit__ = MagicMock(return_value=False)
                return mock_resp

            with patch("urllib.request.urlopen", side_effect=_side_effect):
                result = deliver_event(
                    db, tenant_id=tenant.id, subscription=sub,
                    event_type="message.received", data={},
                    secret=secret, max_attempts=3, base_delay=0,
                )
            assert result.success is True
            assert result.attempts == 2
            # Two log rows: one failure, one success.
            rows = db.query(DeliveryLog).filter(DeliveryLog.tenant_id == tenant.id).all()
            assert len(rows) == 2
        finally:
            db.close()

    def test_all_attempts_exhausted(self):
        """All attempts fail → success is False, attempts == max_attempts."""
        tenant = _make_tenant("wh-exhaust")
        sub, secret = _make_subscription(tenant.id)
        db = SessionLocal()
        try:
            with patch("urllib.request.urlopen", side_effect=ConnectionError("no server")):
                result = deliver_event(
                    db, tenant_id=tenant.id, subscription=sub,
                    event_type="message.received", data={},
                    secret=secret, max_attempts=3, base_delay=0,
                )
            assert result.success is False
            assert result.attempts == 3
            # Three log rows.
            rows = db.query(DeliveryLog).filter(DeliveryLog.tenant_id == tenant.id).all()
            assert len(rows) == 3
        finally:
            db.close()

    def test_no_secret_in_log_rows(self):
        """DeliveryLog rows must not contain the plaintext secret."""
        tenant = _make_tenant("wh-no-secret")
        sub, secret = _make_subscription(tenant.id)
        db = SessionLocal()
        try:
            with patch("urllib.request.urlopen", side_effect=ConnectionError("x")):
                deliver_event(
                    db, tenant_id=tenant.id, subscription=sub,
                    event_type="evt", data={},
                    secret=secret, max_attempts=1, base_delay=0,
                )
            rows = db.query(DeliveryLog).filter(DeliveryLog.tenant_id == tenant.id).all()
            for row in rows:
                for field in (row.error_detail or "", row.payload_preview or ""):
                    assert secret not in field, "plaintext secret found in log row"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# dispatch_event
# ---------------------------------------------------------------------------

class TestDispatchEvent:
    def test_dispatches_to_all_active_subscriptions(self):
        tenant = _make_tenant("wh-dispatch")
        db = SessionLocal()
        try:
            sub1, secret1 = create_subscription(db, tenant_id=tenant.id,
                                                 url="http://a.invalid/hook")
            sub2, secret2 = create_subscription(db, tenant_id=tenant.id,
                                                 url="http://b.invalid/hook")
            secrets_map = {sub1.id: secret1, sub2.id: secret2}

            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)

            with patch("urllib.request.urlopen", return_value=mock_resp):
                results = dispatch_event(
                    db, tenant_id=tenant.id, event_type="message.received",
                    data={"x": 1}, secrets_map=secrets_map, max_attempts=1,
                )
            assert len(results) == 2
            assert all(r.success for r in results)
        finally:
            db.close()

    def test_skips_subscription_without_secret(self):
        tenant = _make_tenant("wh-skip-secret")
        db = SessionLocal()
        try:
            sub, _secret = create_subscription(db, tenant_id=tenant.id,
                                                url="http://c.invalid/hook")
            # Pass empty secrets_map → subscription has no secret available.
            results = dispatch_event(
                db, tenant_id=tenant.id, event_type="message.received",
                data={}, secrets_map={}, max_attempts=1,
            )
            assert len(results) == 0
        finally:
            db.close()

    def test_tenant_isolation(self):
        """Subscriptions for one tenant must not be delivered to another tenant's call."""
        t1 = _make_tenant("wh-iso-t1")
        t2 = _make_tenant("wh-iso-t2")
        db = SessionLocal()
        try:
            sub_t1, secret_t1 = create_subscription(db, tenant_id=t1.id,
                                                      url="http://d.invalid/hook")
            secrets_map = {sub_t1.id: secret_t1}

            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)

            with patch("urllib.request.urlopen", return_value=mock_resp):
                results = dispatch_event(
                    db, tenant_id=t2.id,  # t2 has no subs
                    event_type="message.received",
                    data={}, secrets_map=secrets_map, max_attempts=1,
                )
            assert len(results) == 0
        finally:
            db.close()
