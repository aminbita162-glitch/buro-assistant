"""
Phase 7 operator desk tests.

Covered checklist rows:
  23 – dashboard counts: received, classified, drafted, sent, held, failed
  24 – department queues

All tests use the in-memory SQLite engine from conftest.py.
No external service is contacted.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from app.main import (
    app, Base, engine, SessionLocal,
    Tenant, User, Task,
    hash_password, limiter,
)
from app.ingest.models import Message
from app.ingest.normalize import NormalizedMessage
from app.ingest.ingest import ingest_message
from app.policy.shadow import Draft
from app.policy.approval import ApprovalQueueEntry, enqueue
from app.policy.audit import log_event


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


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tenant(slug: str = "t1") -> Tenant:
    db = SessionLocal()
    try:
        t = Tenant(name=slug, slug=slug)
        db.add(t); db.commit(); db.refresh(t)
        return t
    finally:
        db.close()


def _make_user(tenant: Tenant, email: str = "op@example.com") -> User:
    db = SessionLocal()
    try:
        u = User(
            tenant_id=tenant.id, name="Operator",
            email=email, password_hash=hash_password("pass99"),
        )
        db.add(u); db.commit(); db.refresh(u)
        return u
    finally:
        db.close()


def _login(client, email: str, password: str = "pass99") -> str:
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def _ingest(tenant_id: int, provider_message_id: str, subject: str,
            state_override: str = "new") -> Message:
    """Insert a message row directly (bypasses the HTTP route)."""
    db = SessionLocal()
    try:
        msg = NormalizedMessage(
            provider="fake",
            provider_message_id=provider_message_id,
            tenant_id=tenant_id,
            message_id_header=None,
            subject=subject,
            subject_normalized=NormalizedMessage.normalize_subject(subject),
            sender="s@example.com",
            recipients=[],
            body_text="body",
            attachments=[],
            raw={},
        )
        record, _ = ingest_message(db, msg)
        if state_override != "new":
            record.state = state_override
            db.commit()
            db.refresh(record)
        return record
    finally:
        db.close()


def _make_draft(tenant_id: int, state: str = "draft") -> Draft:
    db = SessionLocal()
    try:
        d = Draft(
            tenant_id=tenant_id,
            subject="Re: test",
            body="Draft body.",
            state=state,
            created_at=datetime.now(timezone.utc),
        )
        db.add(d); db.commit(); db.refresh(d)
        return d
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Auth guard — all desk endpoints require Bearer token
# ---------------------------------------------------------------------------

class TestDeskAuthRequired:
    ENDPOINTS = [
        "/desk/dashboard",
        "/desk/queue",
        "/desk/queue/billing",
        "/desk/decisions",
        "/desk/drafts",
        "/desk/approval",
        "/desk/audit",
        "/desk/tasks",
        "/desk/quota",
        "/desk/cost",
    ]

    def test_all_endpoints_require_auth(self, client):
        for path in self.ENDPOINTS:
            r = client.get(path)
            assert r.status_code == 401, f"{path} returned {r.status_code}, expected 401"

    def test_approve_reject_require_auth(self, client):
        for path in ("/desk/approval/1/approve", "/desk/approval/1/reject"):
            r = client.post(path)
            assert r.status_code == 401, f"{path} returned {r.status_code}, expected 401"


# ---------------------------------------------------------------------------
# Row 23 – Dashboard counts
# ---------------------------------------------------------------------------

class TestDashboard:
    def test_dashboard_returns_200(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        r = client.get("/desk/dashboard", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_dashboard_has_required_keys(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        data = client.get(
            "/desk/dashboard", headers={"Authorization": f"Bearer {token}"}
        ).json()
        assert "messages" in data
        assert "drafts" in data
        assert "held" in data

    def test_dashboard_message_states_present(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        data = client.get(
            "/desk/dashboard", headers={"Authorization": f"Bearer {token}"}
        ).json()
        msgs = data["messages"]
        for key in ("received", "classified", "quarantine", "duplicate", "failed"):
            assert key in msgs, f"missing key: {key}"

    def test_dashboard_draft_states_present(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        data = client.get(
            "/desk/dashboard", headers={"Authorization": f"Bearer {token}"}
        ).json()
        dft = data["drafts"]
        for key in ("drafted", "sent", "approved", "rejected"):
            assert key in dft, f"missing key: {key}"

    def test_received_count_increments(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        hdr = {"Authorization": f"Bearer {token}"}

        before = client.get("/desk/dashboard", headers=hdr).json()["messages"]["received"]
        _ingest(tenant.id, "m1", "New invoice")
        _ingest(tenant.id, "m2", "Another message")
        after = client.get("/desk/dashboard", headers=hdr).json()["messages"]["received"]
        assert after == before + 2

    def test_quarantine_counted_separately(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        hdr = {"Authorization": f"Bearer {token}"}

        _ingest(tenant.id, "q1", "Suspicious file", state_override="quarantine")
        data = client.get("/desk/dashboard", headers=hdr).json()
        assert data["messages"]["quarantine"] >= 1

    def test_drafted_count_includes_shadow(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        hdr = {"Authorization": f"Bearer {token}"}

        _make_draft(tenant.id, "draft")
        _make_draft(tenant.id, "shadow")
        data = client.get("/desk/dashboard", headers=hdr).json()
        assert data["drafts"]["drafted"] >= 2

    def test_held_count_reflects_approval_queue(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        hdr = {"Authorization": f"Bearer {token}"}

        db = SessionLocal()
        try:
            enqueue(db, tenant.id, "Re: invoice", "Please approve.")
        finally:
            db.close()

        data = client.get("/desk/dashboard", headers=hdr).json()
        assert data["held"] >= 1

    def test_dashboard_is_tenant_scoped(self, client):
        """Dashboard must not count other tenants' messages."""
        t1 = _make_tenant("t1-dash")
        t2 = _make_tenant("t2-dash")
        _make_user(t1, "op1@example.com")
        _make_user(t2, "op2@example.com")

        _ingest(t2.id, "other-m1", "Invoice from t2")

        token1 = _login(client, "op1@example.com")
        data = client.get(
            "/desk/dashboard", headers={"Authorization": f"Bearer {token1}"}
        ).json()
        assert data["messages"]["received"] == 0


# ---------------------------------------------------------------------------
# Row 24 – Department queues
# ---------------------------------------------------------------------------

class TestDepartmentQueue:
    def test_queue_all_returns_messages(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        hdr = {"Authorization": f"Bearer {token}"}

        _ingest(tenant.id, "dq-1", "Invoice payment overdue")
        r = client.get("/desk/queue", headers=hdr)
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_department_queue_billing_keyword(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        hdr = {"Authorization": f"Bearer {token}"}

        _ingest(tenant.id, "bill-1", "Invoice payment request")
        _ingest(tenant.id, "hr-1", "Job application engineer")

        r = client.get("/desk/queue/invoice", headers=hdr)
        assert r.status_code == 200
        data = r.json()
        assert data["department"] == "invoice"
        subjects = [m["subject_normalized"] for m in data["messages"]]
        assert any("invoice" in s for s in subjects)

    def test_department_queue_no_match_returns_empty(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        _ingest(tenant.id, "x-1", "Random unrelated message")
        r = client.get(
            "/desk/queue/xyzzy",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_department_queue_is_case_insensitive(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        hdr = {"Authorization": f"Bearer {token}"}

        _ingest(tenant.id, "ci-1", "INVOICE PAYMENT DUE")
        r = client.get("/desk/queue/invoice", headers=hdr)
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_department_queue_is_tenant_scoped(self, client):
        t1 = _make_tenant("t1-dq")
        t2 = _make_tenant("t2-dq")
        _make_user(t1, "dq1@example.com")
        _make_user(t2, "dq2@example.com")

        _ingest(t2.id, "dq-t2", "Invoice from tenant 2")

        token1 = _login(client, "dq1@example.com")
        r = client.get(
            "/desk/queue/invoice",
            headers={"Authorization": f"Bearer {token1}"},
        )
        assert r.json()["count"] == 0

    def test_department_queue_response_shape(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        _ingest(tenant.id, "shape-1", "Support request help needed")
        r = client.get(
            "/desk/queue/support",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "department" in data
        assert "count" in data
        assert "messages" in data
        if data["count"] > 0:
            msg = data["messages"][0]
            for field in ("id", "provider_message_id", "subject_normalized",
                          "state", "attachment_state", "ingest_time", "tenant_id"):
                assert field in msg, f"missing field: {field}"


# ---------------------------------------------------------------------------
# Other desk endpoints — basic smoke tests
# ---------------------------------------------------------------------------

class TestDeskInbound:
    def test_queue_returns_200(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        r = client.get("/desk/queue", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert "messages" in r.json()

    def test_queue_state_filter(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        hdr = {"Authorization": f"Bearer {token}"}
        _ingest(tenant.id, "f1", "New message", state_override="new")
        _ingest(tenant.id, "f2", "Failed message", state_override="failed")
        r = client.get("/desk/queue?state=failed", headers=hdr)
        assert r.status_code == 200
        assert all(m["state"] == "failed" for m in r.json()["messages"])


class TestDeskDecisions:
    def test_decisions_returns_200(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        r = client.get("/desk/decisions", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert "decisions" in r.json()


class TestDeskDrafts:
    def test_drafts_returns_200(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        r = client.get("/desk/drafts", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert "drafts" in r.json()

    def test_drafts_shows_stored_draft(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        _make_draft(tenant.id, "draft")
        r = client.get("/desk/drafts", headers={"Authorization": f"Bearer {token}"})
        assert r.json()["count"] >= 1

    def test_drafts_state_filter(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        hdr = {"Authorization": f"Bearer {token}"}
        _make_draft(tenant.id, "draft")
        _make_draft(tenant.id, "shadow")
        r = client.get("/desk/drafts?state=shadow", headers=hdr)
        assert r.status_code == 200
        assert all(d["state"] == "shadow" for d in r.json()["drafts"])


class TestDeskApproval:
    def test_approval_returns_200(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        r = client.get("/desk/approval", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert "entries" in r.json()

    def test_approval_shows_pending_entry(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        db = SessionLocal()
        try:
            enqueue(db, tenant.id, "Re: subject", "Body text.")
        finally:
            db.close()
        r = client.get("/desk/approval", headers={"Authorization": f"Bearer {token}"})
        assert r.json()["count"] >= 1


class TestDeskAudit:
    def test_audit_returns_200(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        r = client.get("/desk/audit", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert "entries" in r.json()

    def test_audit_shows_logged_event(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        db = SessionLocal()
        try:
            log_event(db, tenant.id, "test_event", actor="system")
        finally:
            db.close()
        r = client.get("/desk/audit", headers={"Authorization": f"Bearer {token}"})
        events = [e["event"] for e in r.json()["entries"]]
        assert "test_event" in events


class TestDeskTasks:
    def test_tasks_returns_200(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        r = client.get("/desk/tasks", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert "tasks" in r.json()

    def test_tasks_shows_active_task(self, client):
        tenant = _make_tenant("default")
        user = _make_user(tenant)
        token = _login(client, "op@example.com")
        db = SessionLocal()
        try:
            t = Task(
                tenant_id=tenant.id, user_id=user.id,
                title="Review invoice", deadline="2026-12-01",
                priority="high", status="active",
            )
            db.add(t); db.commit()
        finally:
            db.close()
        r = client.get("/desk/tasks", headers={"Authorization": f"Bearer {token}"})
        assert r.json()["count"] >= 1

    def test_desk_tasks_excludes_completed(self, client):
        tenant = _make_tenant("default")
        user = _make_user(tenant)
        token = _login(client, "op@example.com")
        db = SessionLocal()
        try:
            db.add(Task(tenant_id=tenant.id, user_id=user.id,
                        title="Done task", deadline="", priority="low",
                        status="completed"))
            db.commit()
        finally:
            db.close()
        r = client.get("/desk/tasks", headers={"Authorization": f"Bearer {token}"})
        assert r.json()["count"] == 0


# ---------------------------------------------------------------------------
# Phase 3 – Approve / reject
# ---------------------------------------------------------------------------

class TestApproveReject:
    def test_approve_returns_200(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        db = SessionLocal()
        try:
            entry = enqueue(db, tenant.id, "Re: invoice", "Please approve.")
        finally:
            db.close()
        r = client.post(
            f"/desk/approval/{entry.id}/approve",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "approved"

    def test_reject_returns_200(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        db = SessionLocal()
        try:
            entry = enqueue(db, tenant.id, "Re: invoice", "Please reject.")
        finally:
            db.close()
        r = client.post(
            f"/desk/approval/{entry.id}/reject",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "rejected"

    def test_approve_missing_entry_returns_404(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        r = client.post(
            "/desk/approval/9999/approve",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 404

    def test_reject_missing_entry_returns_404(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        r = client.post(
            "/desk/approval/9999/reject",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 404

    def test_approve_cross_tenant_returns_404(self, client):
        """Operator from tenant 1 cannot approve tenant 2's entry."""
        t1 = _make_tenant("t1-apr")
        t2 = _make_tenant("t2-apr")
        _make_user(t1, "apr1@example.com")
        _make_user(t2, "apr2@example.com")
        db = SessionLocal()
        try:
            entry = enqueue(db, t2.id, "Re: invoice", "Tenant 2 approval.")
        finally:
            db.close()
        token1 = _login(client, "apr1@example.com")
        r = client.post(
            f"/desk/approval/{entry.id}/approve",
            headers={"Authorization": f"Bearer {token1}"},
        )
        assert r.status_code == 404

    def test_approve_twice_returns_404(self, client):
        """Approving an already-resolved entry returns 404."""
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        hdr = {"Authorization": f"Bearer {token}"}
        db = SessionLocal()
        try:
            entry = enqueue(db, tenant.id, "Re: double", "Body.")
        finally:
            db.close()
        r1 = client.post(f"/desk/approval/{entry.id}/approve", headers=hdr)
        assert r1.status_code == 200
        r2 = client.post(f"/desk/approval/{entry.id}/approve", headers=hdr)
        assert r2.status_code == 404

    def test_entry_not_in_approval_list_after_resolve(self, client):
        """After resolving, entry no longer appears in pending list."""
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        hdr = {"Authorization": f"Bearer {token}"}
        db = SessionLocal()
        try:
            entry = enqueue(db, tenant.id, "Re: remove", "Body.")
        finally:
            db.close()
        client.post(f"/desk/approval/{entry.id}/approve", headers=hdr)
        r = client.get("/desk/approval", headers=hdr)
        ids = [e["id"] for e in r.json()["entries"]]
        assert entry.id not in ids


# ---------------------------------------------------------------------------
# Phase 3 – Quota
# ---------------------------------------------------------------------------

class TestDeskQuota:
    def test_quota_returns_200(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        r = client.get("/desk/quota", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_quota_has_required_keys(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        data = client.get(
            "/desk/quota", headers={"Authorization": f"Bearer {token}"}
        ).json()
        for key in ("quota_date", "daily_token_quota", "tokens_used",
                    "tokens_remaining", "cost_usd_used"):
            assert key in data, f"missing key: {key}"

    def test_quota_is_tenant_scoped(self, client):
        t1 = _make_tenant("t1-quota")
        t2 = _make_tenant("t2-quota")
        _make_user(t1, "q1@example.com")
        _make_user(t2, "q2@example.com")
        token1 = _login(client, "q1@example.com")
        data = client.get(
            "/desk/quota", headers={"Authorization": f"Bearer {token1}"}
        ).json()
        assert data["tenant_id"] == t1.id

    def test_quota_tokens_remaining_not_negative(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        data = client.get(
            "/desk/quota", headers={"Authorization": f"Bearer {token}"}
        ).json()
        assert data["tokens_remaining"] >= 0


# ---------------------------------------------------------------------------
# Phase 3 – Cost
# ---------------------------------------------------------------------------

class TestDeskCost:
    def test_cost_returns_200(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        r = client.get("/desk/cost", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_cost_has_required_keys(self, client):
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        data = client.get(
            "/desk/cost", headers={"Authorization": f"Bearer {token}"}
        ).json()
        for key in ("total_tokens", "total_cost_usd", "event_count", "events"):
            assert key in data, f"missing key: {key}"

    def test_cost_is_tenant_scoped(self, client):
        t1 = _make_tenant("t1-cost")
        t2 = _make_tenant("t2-cost")
        _make_user(t1, "c1@example.com")
        _make_user(t2, "c2@example.com")
        token1 = _login(client, "c1@example.com")
        data = client.get(
            "/desk/cost", headers={"Authorization": f"Bearer {token1}"}
        ).json()
        assert data["tenant_id"] == t1.id


# ---------------------------------------------------------------------------
# Phase 3 – Department as stored field
# ---------------------------------------------------------------------------

class TestDepartmentStoredField:
    def test_message_serialiser_includes_department(self, client):
        """Queue response includes 'department' in each message dict."""
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        _ingest(tenant.id, "dept-1", "Billing request")
        r = client.get("/desk/queue", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        messages = r.json()["messages"]
        assert len(messages) >= 1
        assert "department" in messages[0]

    def test_department_queue_matches_stored_field(self, client):
        """
        When a message has department='support' stored, it is returned by
        /desk/queue/support even if the subject does not contain 'support'.
        """
        tenant = _make_tenant("default")
        _make_user(tenant)
        token = _login(client, "op@example.com")
        hdr = {"Authorization": f"Bearer {token}"}

        # Insert message with explicit department column.
        db = SessionLocal()
        try:
            from app.ingest.normalize import NormalizedMessage
            from app.ingest.ingest import ingest_message
            msg = NormalizedMessage(
                provider="fake",
                provider_message_id="dept-stored-1",
                tenant_id=tenant.id,
                message_id_header=None,
                subject="Completely unrelated title",
                subject_normalized="completely unrelated title",
                sender="s@example.com",
                recipients=[],
                body_text="body",
                attachments=[],
                raw={},
            )
            record, _ = ingest_message(db, msg)
            record.department = "support"
            db.commit()
        finally:
            db.close()

        r = client.get("/desk/queue/support", headers=hdr)
        assert r.status_code == 200
        data = r.json()
        subjects = [m["subject_normalized"] for m in data["messages"]]
        assert any("unrelated" in s for s in subjects), (
            "Message with stored department='support' not found in /desk/queue/support"
        )
