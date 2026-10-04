"""
app/domain/buyer_features.py – fifteen buyer-option features (Phase 7 / Section B).

Each feature has its own ORM model (where persistence is needed) and a thin
helper API.  Features that are purely policy-layer use no new table but add a
helper function that the existing models already support.

Section B items implemented here
---------------------------------
B1   Shared team inbox with assignment          → MessageAssignment table
B2   Internal note (never sent)                 → InternalNote table
B3   Collision lock for drafts                  → DraftLock table
B4   Snooze until a business hour               → SnoozedMessage table
B5   VIP sender list per tenant                 → VipSender table
B6   After-hours holding queue                  → uses SnoozedMessage (after-hours flag)
B7   Saved reply snippets (template-checked)    → ReplySnippet table
B8   Per-department SLA                         → DepartmentSla table
B9   CSV export of audit log                    → csv_export_audit() helper
B10  Bounce and failure reason on desk          → DeliveryFailure table
B11  Vacation responder (no invented facts)     → VacationResponder table
B12  Legal hold that blocks retention delete    → already on Message.legal_hold;
                                                   set_legal_hold() helper added here
B13  Role split: admin, operator, auditor       → UserRole table
B14  German and English receipt templates       → register_language_templates() helper
B15  Daily operator digest, shadow by default   → DigestSubscription table +
                                                   build_digest() helper

All ORM models live here; migration 0010 creates the new tables.
All helpers are pure functions (no side effects beyond the DB session passed in).
None of these helpers send mail or call a model.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Session

from app.main import Base


# ===========================================================================
# B1 – Shared team inbox with assignment
# ===========================================================================

class MessageAssignment(Base):
    """
    Records which operator a message is assigned to inside the shared inbox.

    One row per (tenant_id, message_id) — reassignment overwrites the row.
    """
    __tablename__ = "message_assignments"

    id             = Column(Integer, primary_key=True, index=True)
    tenant_id      = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    message_id     = Column(Integer, ForeignKey("messages.id"), nullable=False, index=True)
    assigned_to    = Column(String, nullable=False)   # user email or user_id string
    assigned_by    = Column(String, nullable=True)    # operator who made the assignment
    assigned_at    = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "message_id", name="uq_assignment_tenant_message"),
    )


def assign_message(
    db: Session,
    tenant_id: int,
    message_id: int,
    assigned_to: str,
    assigned_by: str = "system",
) -> MessageAssignment:
    """Assign or reassign a message to an operator."""
    existing = (
        db.query(MessageAssignment)
        .filter(
            MessageAssignment.tenant_id == tenant_id,
            MessageAssignment.message_id == message_id,
        )
        .first()
    )
    if existing:
        existing.assigned_to = assigned_to
        existing.assigned_by = assigned_by
        existing.assigned_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(existing)
        return existing

    row = MessageAssignment(
        tenant_id=tenant_id,
        message_id=message_id,
        assigned_to=assigned_to,
        assigned_by=assigned_by,
        assigned_at=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_assignment(
    db: Session, tenant_id: int, message_id: int
) -> Optional[MessageAssignment]:
    return (
        db.query(MessageAssignment)
        .filter(
            MessageAssignment.tenant_id == tenant_id,
            MessageAssignment.message_id == message_id,
        )
        .first()
    )


# ===========================================================================
# B2 – Internal note (never sent)
# ===========================================================================

class InternalNote(Base):
    """
    Operator notes attached to a message.  Never included in outbound mail.
    """
    __tablename__ = "internal_notes"

    id          = Column(Integer, primary_key=True, index=True)
    tenant_id   = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    message_id  = Column(Integer, ForeignKey("messages.id"), nullable=False, index=True)
    author      = Column(String, nullable=False)      # user email
    body        = Column(Text, nullable=False)
    created_at  = Column(DateTime(timezone=True), nullable=False)


def add_internal_note(
    db: Session,
    tenant_id: int,
    message_id: int,
    author: str,
    body: str,
) -> InternalNote:
    """Add an internal note to a message.  The note is never sent."""
    note = InternalNote(
        tenant_id=tenant_id,
        message_id=message_id,
        author=author,
        body=body,
        created_at=datetime.now(timezone.utc),
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


def notes_for_message(
    db: Session, tenant_id: int, message_id: int
) -> List[InternalNote]:
    return (
        db.query(InternalNote)
        .filter(
            InternalNote.tenant_id == tenant_id,
            InternalNote.message_id == message_id,
        )
        .order_by(InternalNote.created_at.asc())
        .all()
    )


# ===========================================================================
# B3 – Collision lock when two operators open one draft
# ===========================================================================

LOCK_TTL_SECONDS = 120   # lock expires after 2 minutes of inactivity


class DraftLock(Base):
    """
    Prevents two operators from editing the same draft simultaneously.

    A lock is acquired by calling ``acquire_draft_lock``.  It expires after
    LOCK_TTL_SECONDS and must be refreshed to stay active.
    """
    __tablename__ = "draft_locks"

    id          = Column(Integer, primary_key=True, index=True)
    tenant_id   = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    draft_id    = Column(Integer, ForeignKey("drafts.id"), nullable=False, index=True)
    locked_by   = Column(String, nullable=False)   # user email
    locked_at   = Column(DateTime(timezone=True), nullable=False)
    expires_at  = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "draft_id", name="uq_lock_tenant_draft"),
    )


def acquire_draft_lock(
    db: Session,
    tenant_id: int,
    draft_id: int,
    user_email: str,
    ttl_seconds: int = LOCK_TTL_SECONDS,
) -> bool:
    """
    Try to acquire an exclusive lock on a draft.

    Returns True if the lock was acquired (or the caller already holds it).
    Returns False if another operator currently holds a live lock.
    """
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    expires = now + timedelta(seconds=ttl_seconds)

    existing = (
        db.query(DraftLock)
        .filter(
            DraftLock.tenant_id == tenant_id,
            DraftLock.draft_id == draft_id,
        )
        .first()
    )

    if existing:
        existing_expires = existing.expires_at
        if existing_expires is not None and existing_expires.tzinfo is None:
            existing_expires = existing_expires.replace(tzinfo=timezone.utc)
        # Lock held by someone else and still live → deny.
        if existing.locked_by != user_email and existing_expires is not None and existing_expires > now:
            return False
        # Lock expired or held by same user → refresh.
        existing.locked_by = user_email
        existing.locked_at = now
        existing.expires_at = expires
        db.commit()
        return True

    lock = DraftLock(
        tenant_id=tenant_id,
        draft_id=draft_id,
        locked_by=user_email,
        locked_at=now,
        expires_at=expires,
    )
    db.add(lock)
    db.commit()
    return True


def release_draft_lock(
    db: Session, tenant_id: int, draft_id: int, user_email: str
) -> bool:
    """Release a lock held by user_email.  Returns True if released."""
    existing = (
        db.query(DraftLock)
        .filter(
            DraftLock.tenant_id == tenant_id,
            DraftLock.draft_id == draft_id,
            DraftLock.locked_by == user_email,
        )
        .first()
    )
    if not existing:
        return False
    db.delete(existing)
    db.commit()
    return True


def get_draft_lock(
    db: Session, tenant_id: int, draft_id: int
) -> Optional[DraftLock]:
    """Return the current live lock for a draft, or None."""
    now = datetime.now(timezone.utc)
    lock = (
        db.query(DraftLock)
        .filter(
            DraftLock.tenant_id == tenant_id,
            DraftLock.draft_id == draft_id,
        )
        .first()
    )
    if lock:
        lock_expires = lock.expires_at
        if lock_expires is not None and lock_expires.tzinfo is None:
            lock_expires = lock_expires.replace(tzinfo=timezone.utc)
        if lock_expires is not None and lock_expires <= now:
            db.delete(lock)
            db.commit()
            return None
    return lock


# ===========================================================================
# B4 – Snooze until a business hour  (also used by B6)
# ===========================================================================

class SnoozedMessage(Base):
    """
    A snoozed or after-hours-held message resurfaces when the snooze expires.

    ``reason`` distinguishes user-triggered snooze (``snooze``) from the
    after-hours holding queue (``after_hours``).
    """
    __tablename__ = "snoozed_messages"

    id          = Column(Integer, primary_key=True, index=True)
    tenant_id   = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    message_id  = Column(Integer, ForeignKey("messages.id"), nullable=False, index=True)
    wake_at     = Column(DateTime(timezone=True), nullable=False, index=True)
    snoozed_by  = Column(String, nullable=True)
    reason      = Column(String, nullable=False, default="snooze")  # snooze | after_hours
    created_at  = Column(DateTime(timezone=True), nullable=False)


def snooze_message(
    db: Session,
    tenant_id: int,
    message_id: int,
    wake_at: datetime,
    snoozed_by: str = "system",
    reason: str = "snooze",
) -> SnoozedMessage:
    """Snooze a message until wake_at."""
    row = SnoozedMessage(
        tenant_id=tenant_id,
        message_id=message_id,
        wake_at=wake_at,
        snoozed_by=snoozed_by,
        reason=reason,
        created_at=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def due_snoozed_messages(
    db: Session,
    tenant_id: int,
    now: Optional[datetime] = None,
) -> List[SnoozedMessage]:
    """Return snooze rows whose wake_at has passed (ready to resurface)."""
    if now is None:
        now = datetime.now(timezone.utc)
    return (
        db.query(SnoozedMessage)
        .filter(
            SnoozedMessage.tenant_id == tenant_id,
            SnoozedMessage.wake_at <= now,
        )
        .order_by(SnoozedMessage.wake_at.asc())
        .all()
    )


# ===========================================================================
# B5 – VIP sender list per tenant
# ===========================================================================

class VipSender(Base):
    """
    An email address or domain that is treated as VIP for a tenant.

    VIP messages may get a different SLA tier or automatic high-urgency
    classification.
    """
    __tablename__ = "vip_senders"

    id          = Column(Integer, primary_key=True, index=True)
    tenant_id   = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    pattern     = Column(String, nullable=False)  # exact address or @domain
    label       = Column(String, nullable=True)   # display label
    added_by    = Column(String, nullable=True)
    added_at    = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "pattern", name="uq_vip_tenant_pattern"),
    )


def add_vip_sender(
    db: Session,
    tenant_id: int,
    pattern: str,
    label: Optional[str] = None,
    added_by: str = "system",
) -> VipSender:
    row = VipSender(
        tenant_id=tenant_id,
        pattern=pattern.strip().lower(),
        label=label,
        added_by=added_by,
        added_at=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def is_vip_sender(
    db: Session, tenant_id: int, sender_email: str
) -> bool:
    """
    Return True if sender_email matches any VIP pattern for the tenant.

    Checks exact address match first, then @domain suffix match.
    """
    sender_lower = sender_email.strip().lower()
    patterns = [
        r.pattern
        for r in db.query(VipSender)
        .filter(VipSender.tenant_id == tenant_id)
        .all()
    ]
    for p in patterns:
        if p == sender_lower:
            return True
        if p.startswith("@") and sender_lower.endswith(p):
            return True
    return False


# ===========================================================================
# B6 – After-hours holding queue
# Uses SnoozedMessage with reason="after_hours".  See also B4.
# ===========================================================================

def hold_after_hours(
    db: Session,
    tenant_id: int,
    message_id: int,
    next_business_start: datetime,
) -> SnoozedMessage:
    """
    Place a message in the after-hours holding queue until the next
    business-hour start.

    This is a convenience wrapper around ``snooze_message`` with
    reason="after_hours".
    """
    return snooze_message(
        db,
        tenant_id=tenant_id,
        message_id=message_id,
        wake_at=next_business_start,
        snoozed_by="system",
        reason="after_hours",
    )


# ===========================================================================
# B7 – Saved reply snippets (template-checked)
# ===========================================================================

class ReplySnippet(Base):
    """
    A short reusable text block an operator can insert into a draft reply.

    Before use the snippet body is validated through
    ``check_forbidden_phrases`` from the template registry.
    """
    __tablename__ = "reply_snippets"

    id          = Column(Integer, primary_key=True, index=True)
    tenant_id   = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    title       = Column(String, nullable=False)
    body        = Column(Text, nullable=False)
    created_by  = Column(String, nullable=True)
    created_at  = Column(DateTime(timezone=True), nullable=False)


def save_reply_snippet(
    db: Session,
    tenant_id: int,
    title: str,
    body: str,
    created_by: str = "system",
    forbidden_phrases: Optional[frozenset] = None,
) -> ReplySnippet:
    """
    Save a reply snippet after template-checking the body.

    Raises ValueError if the body contains any forbidden phrase.
    """
    if forbidden_phrases:
        from app.policy.templates import check_forbidden_phrases
        hits = check_forbidden_phrases(body, forbidden_phrases)
        if hits:
            raise ValueError(f"Snippet body contains forbidden phrases: {hits}")

    row = ReplySnippet(
        tenant_id=tenant_id,
        title=title,
        body=body,
        created_by=created_by,
        created_at=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def list_snippets(db: Session, tenant_id: int) -> List[ReplySnippet]:
    return (
        db.query(ReplySnippet)
        .filter(ReplySnippet.tenant_id == tenant_id)
        .order_by(ReplySnippet.created_at.asc())
        .all()
    )


# ===========================================================================
# B8 – Per-department SLA
# ===========================================================================

class DepartmentSla(Base):
    """
    SLA hours overrides per department per tenant.

    Takes precedence over the global SLA config when a message has a
    matching stored department value.
    """
    __tablename__ = "department_sla"

    id             = Column(Integer, primary_key=True, index=True)
    tenant_id      = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    department     = Column(String, nullable=False)
    critical_hours = Column(Integer, nullable=False, default=1)
    high_hours     = Column(Integer, nullable=False, default=4)
    medium_hours   = Column(Integer, nullable=False, default=24)
    low_hours      = Column(Integer, nullable=False, default=72)

    __table_args__ = (
        UniqueConstraint("tenant_id", "department", name="uq_dept_sla_tenant_dept"),
    )


def set_department_sla(
    db: Session,
    tenant_id: int,
    department: str,
    critical_hours: int = 1,
    high_hours: int = 4,
    medium_hours: int = 24,
    low_hours: int = 72,
) -> DepartmentSla:
    """Create or update the SLA config for a department."""
    existing = (
        db.query(DepartmentSla)
        .filter(
            DepartmentSla.tenant_id == tenant_id,
            DepartmentSla.department == department.strip().lower(),
        )
        .first()
    )
    if existing:
        existing.critical_hours = critical_hours
        existing.high_hours = high_hours
        existing.medium_hours = medium_hours
        existing.low_hours = low_hours
        db.commit()
        db.refresh(existing)
        return existing

    row = DepartmentSla(
        tenant_id=tenant_id,
        department=department.strip().lower(),
        critical_hours=critical_hours,
        high_hours=high_hours,
        medium_hours=medium_hours,
        low_hours=low_hours,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_department_sla_config(
    db: Session, tenant_id: int, department: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    Return a sla_config dict compatible with ``policy.sla.sla_deadline``,
    or None if no department SLA is configured.
    """
    if not department:
        return None
    row = (
        db.query(DepartmentSla)
        .filter(
            DepartmentSla.tenant_id == tenant_id,
            DepartmentSla.department == department.strip().lower(),
        )
        .first()
    )
    if not row:
        return None
    return {
        "sla_hours": {
            "critical": row.critical_hours,
            "high": row.high_hours,
            "medium": row.medium_hours,
            "low": row.low_hours,
        }
    }


# ===========================================================================
# B9 – CSV export of audit log
# ===========================================================================

def csv_export_audit(db: Session, tenant_id: int, limit: int = 10_000) -> str:
    """
    Return the audit log for *tenant_id* as a UTF-8 CSV string.

    Columns: id, event, actor, message_id, detail, created_at
    """
    from app.policy.audit import recent_for_tenant

    rows = recent_for_tenant(db, tenant_id, limit=limit)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "event", "actor", "message_id", "detail", "created_at"])
    for e in rows:
        writer.writerow([
            e.id,
            e.event,
            e.actor or "",
            e.message_id or "",
            e.detail or "",
            e.created_at.isoformat() if e.created_at else "",
        ])
    return buf.getvalue()


# ===========================================================================
# B10 – Bounce and failure reason on the desk
# ===========================================================================

class DeliveryFailure(Base):
    """
    Records a bounced or failed outbound delivery attempt.

    The ``reason`` and ``smtp_code`` fields surface on the operator desk
    so failures can be investigated without digging through mail server logs.
    """
    __tablename__ = "delivery_failures"

    id           = Column(Integer, primary_key=True, index=True)
    tenant_id    = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    message_id   = Column(Integer, ForeignKey("messages.id"), nullable=True, index=True)
    draft_id     = Column(Integer, ForeignKey("drafts.id"), nullable=True, index=True)
    recipient    = Column(String, nullable=True)
    smtp_code    = Column(Integer, nullable=True)   # e.g. 550, 421
    reason       = Column(Text, nullable=True)      # human-readable failure reason
    failed_at    = Column(DateTime(timezone=True), nullable=False)


def record_delivery_failure(
    db: Session,
    tenant_id: int,
    reason: str,
    message_id: Optional[int] = None,
    draft_id: Optional[int] = None,
    recipient: Optional[str] = None,
    smtp_code: Optional[int] = None,
) -> DeliveryFailure:
    row = DeliveryFailure(
        tenant_id=tenant_id,
        message_id=message_id,
        draft_id=draft_id,
        recipient=recipient,
        smtp_code=smtp_code,
        reason=reason,
        failed_at=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def failures_for_tenant(
    db: Session, tenant_id: int, limit: int = 100
) -> List[DeliveryFailure]:
    return (
        db.query(DeliveryFailure)
        .filter(DeliveryFailure.tenant_id == tenant_id)
        .order_by(DeliveryFailure.failed_at.desc())
        .limit(limit)
        .all()
    )


# ===========================================================================
# B11 – Vacation responder that cannot invent facts
# ===========================================================================

class VacationResponder(Base):
    """
    A tenant-level vacation auto-responder.

    The ``template_id`` must resolve to a registered template so the reply
    is rendered from fixed variables — no model call, no invented facts.

    ``active`` is false by default; the operator must explicitly enable it.
    """
    __tablename__ = "vacation_responders"

    id            = Column(Integer, primary_key=True, index=True)
    tenant_id     = Column(Integer, ForeignKey("tenants.id"), nullable=False, unique=True, index=True)
    active        = Column(Boolean, nullable=False, default=False)
    template_id   = Column(String, nullable=False)    # must be in template registry
    start_date    = Column(DateTime(timezone=True), nullable=True)
    end_date      = Column(DateTime(timezone=True), nullable=True)
    created_by    = Column(String, nullable=True)
    updated_at    = Column(DateTime(timezone=True), nullable=False)


def set_vacation_responder(
    db: Session,
    tenant_id: int,
    template_id: str,
    active: bool = False,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    created_by: str = "system",
) -> VacationResponder:
    """Create or update the vacation responder for a tenant."""
    existing = (
        db.query(VacationResponder)
        .filter(VacationResponder.tenant_id == tenant_id)
        .first()
    )
    now = datetime.now(timezone.utc)
    if existing:
        existing.template_id = template_id
        existing.active = active
        existing.start_date = start_date
        existing.end_date = end_date
        existing.updated_at = now
        db.commit()
        db.refresh(existing)
        return existing

    row = VacationResponder(
        tenant_id=tenant_id,
        template_id=template_id,
        active=active,
        start_date=start_date,
        end_date=end_date,
        created_by=created_by,
        updated_at=now,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_vacation_responder(
    db: Session, tenant_id: int
) -> Optional[VacationResponder]:
    return (
        db.query(VacationResponder)
        .filter(VacationResponder.tenant_id == tenant_id)
        .first()
    )


def vacation_responder_active(
    db: Session, tenant_id: int, now: Optional[datetime] = None
) -> bool:
    """
    Return True if the vacation responder is active for *tenant_id* at *now*.

    Checks ``active`` flag, and optional start/end window.
    A vacation responder never calls a model; it can only return a template reply.
    """
    row = get_vacation_responder(db, tenant_id)
    if not row or not row.active:
        return False
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if row.start_date:
        start = row.start_date
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if now < start:
            return False
    if row.end_date:
        end = row.end_date
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        if now > end:
            return False
    return True


# ===========================================================================
# B12 – Legal hold that blocks retention delete
# Already implemented as Message.legal_hold (migration 0009).
# This helper adds the API-facing set/clear functions.
# ===========================================================================

def set_legal_hold(
    db: Session,
    tenant_id: int,
    message_id: int,
    hold: bool,
) -> bool:
    """
    Set or clear the legal hold flag on a message.

    Returns True if the message was found and updated, False if not found
    or belongs to a different tenant.
    """
    from app.ingest.models import Message

    msg = (
        db.query(Message)
        .filter(Message.id == message_id, Message.tenant_id == tenant_id)
        .first()
    )
    if not msg:
        return False
    msg.legal_hold = hold
    db.commit()
    return True


# ===========================================================================
# B13 – Role split: admin, operator, auditor
# ===========================================================================

VALID_ROLES = frozenset({"admin", "operator", "auditor"})


class UserRole(Base):
    """
    Per-tenant role assignment for a user.

    role values: admin | operator | auditor

    - admin     : full access including privacy endpoints and role management
    - operator  : can approve/reject drafts and view all queues
    - auditor   : read-only access to audit log and dashboard
    """
    __tablename__ = "user_roles"

    id         = Column(Integer, primary_key=True, index=True)
    tenant_id  = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    role       = Column(String, nullable=False, default="operator")
    granted_by = Column(String, nullable=True)
    granted_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "user_id", name="uq_role_tenant_user"),
    )


def set_user_role(
    db: Session,
    tenant_id: int,
    user_id: int,
    role: str,
    granted_by: str = "system",
) -> UserRole:
    """Assign or update the role for a user within a tenant."""
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role {role!r}. Must be one of {sorted(VALID_ROLES)}")

    existing = (
        db.query(UserRole)
        .filter(
            UserRole.tenant_id == tenant_id,
            UserRole.user_id == user_id,
        )
        .first()
    )
    if existing:
        existing.role = role
        existing.granted_by = granted_by
        existing.granted_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(existing)
        return existing

    row = UserRole(
        tenant_id=tenant_id,
        user_id=user_id,
        role=role,
        granted_by=granted_by,
        granted_at=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_user_role(
    db: Session, tenant_id: int, user_id: int
) -> Optional[str]:
    """Return the user's role string, or None if no role is set."""
    row = (
        db.query(UserRole)
        .filter(
            UserRole.tenant_id == tenant_id,
            UserRole.user_id == user_id,
        )
        .first()
    )
    return row.role if row else None


# ===========================================================================
# B14 – German and English receipt templates
# ===========================================================================

def register_language_templates(registry) -> None:
    """
    Register German (de) and English (en) receipt templates in *registry*.

    Both templates satisfy the same constraint as the built-in receipt:
    they state that the message was received and will be reviewed.
    No model is called; templates have no free-form generation.
    """
    from app.policy.templates import TemplateEntry

    _FORBIDDEN = frozenset([
        "guarantee", "warranty", "liability", "no refund", "not responsible",
        "garantie", "gewährleistung", "haftung", "keine erstattung",
    ])

    en = TemplateEntry(
        id="receipt-en",
        subject="Re: {original_subject}",
        body=(
            "Dear {sender_name},\n\n"
            "We have received your message and it will be reviewed by our team.\n\n"
            "Reference: {reference}\n\n"
            "Thank you,\n"
            "The Buro Team"
        ),
        required_variables={"original_subject", "sender_name", "reference"},
        forbidden_phrases=_FORBIDDEN,
        language="en",
    )

    de = TemplateEntry(
        id="receipt-de",
        subject="Re: {original_subject}",
        body=(
            "Sehr geehrte/r {sender_name},\n\n"
            "Wir haben Ihre Nachricht erhalten und sie wird von unserem Team geprüft.\n\n"
            "Referenz: {reference}\n\n"
            "Vielen Dank,\n"
            "Das Buro-Team"
        ),
        required_variables={"original_subject", "sender_name", "reference"},
        forbidden_phrases=_FORBIDDEN,
        language="de",
    )

    registry.register(en)
    registry.register(de)


# ===========================================================================
# B15 – Daily operator digest, shadow by default
# ===========================================================================

class DigestSubscription(Base):
    """
    Controls whether and how an operator receives a daily digest.

    ``send_mode`` values:
      shadow  – digest is built and stored but not sent (default)
      email   – digest is sent to ``recipient_email``
    """
    __tablename__ = "digest_subscriptions"

    id               = Column(Integer, primary_key=True, index=True)
    tenant_id        = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    recipient_email  = Column(String, nullable=False)
    send_mode        = Column(String, nullable=False, default="shadow")  # shadow | email
    active           = Column(Boolean, nullable=False, default=True)
    created_at       = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "user_id", name="uq_digest_tenant_user"),
    )


def set_digest_subscription(
    db: Session,
    tenant_id: int,
    user_id: int,
    recipient_email: str,
    send_mode: str = "shadow",
    active: bool = True,
) -> DigestSubscription:
    """Create or update a digest subscription for a user."""
    if send_mode not in ("shadow", "email"):
        raise ValueError(f"Invalid send_mode {send_mode!r}. Must be 'shadow' or 'email'.")

    existing = (
        db.query(DigestSubscription)
        .filter(
            DigestSubscription.tenant_id == tenant_id,
            DigestSubscription.user_id == user_id,
        )
        .first()
    )
    if existing:
        existing.recipient_email = recipient_email
        existing.send_mode = send_mode
        existing.active = active
        db.commit()
        db.refresh(existing)
        return existing

    row = DigestSubscription(
        tenant_id=tenant_id,
        user_id=user_id,
        recipient_email=recipient_email,
        send_mode=send_mode,
        active=active,
        created_at=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def build_digest(db: Session, tenant_id: int) -> Dict[str, Any]:
    """
    Build the daily digest payload for a tenant.

    Returns a dict with counts and the most recent decisions, drafts, and
    failures.  No model is called.  Shadow-mode subscriptions store this
    dict without emailing it.
    """
    from app.ingest.models import Message
    from app.policy.shadow import Draft
    from app.policy.approval import ApprovalQueueEntry

    # Counts
    received = db.query(Message).filter(Message.tenant_id == tenant_id).count()
    drafted  = db.query(Draft).filter(Draft.tenant_id == tenant_id).count()
    held     = (
        db.query(ApprovalQueueEntry)
        .filter(
            ApprovalQueueEntry.tenant_id == tenant_id,
            ApprovalQueueEntry.state == "pending",
        )
        .count()
    )
    failures = db.query(DeliveryFailure).filter(
        DeliveryFailure.tenant_id == tenant_id
    ).count()

    return {
        "tenant_id": tenant_id,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "messages_received": received,
        "drafts_created": drafted,
        "held_for_approval": held,
        "delivery_failures": failures,
    }
