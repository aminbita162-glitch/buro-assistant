from __future__ import annotations

import json
import os
import hashlib
import secrets
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError

from fastapi import FastAPI, HTTPException, Query, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, EmailStr
from openai import OpenAI

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, ForeignKey, or_, text, inspect,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

_ph = PasswordHasher()

DATABASE_URL = os.getenv("DATABASE_URL")

_engine_kwargs: dict = {"pool_pre_ping": True}
if DATABASE_URL and not DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["pool_recycle"] = 300
    _engine_kwargs["pool_timeout"] = 30

engine = create_engine(DATABASE_URL, **_engine_kwargs)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

SESSION_TTL_HOURS = 24


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    slug = Column(String, nullable=False, unique=True, index=True)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    last_task_id = Column(Integer, nullable=True)


class UserSession(Base):
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    token_hash = Column(String, unique=True, index=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=True)
    title = Column(String)
    deadline = Column(String)
    priority = Column(String)
    status = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# Schema — managed by Alembic only.  No create_all or ALTER TABLE here.
# The startup function below runs "alembic upgrade head" once so existing
# databases are migrated automatically when the process starts.
# ---------------------------------------------------------------------------

def _run_migrations() -> None:
    """Apply pending Alembic migrations at startup."""
    from alembic.config import Config
    from alembic import command as alembic_command
    cfg = Config("alembic.ini")
    alembic_command.upgrade(cfg, "head")


try:
    _run_migrations()
except Exception as _mig_err:  # noqa: BLE001
    # Log but do not crash — the process still starts; the operator can
    # run `alembic upgrade head` manually if the DB is not reachable yet.
    import logging as _logging
    _logging.getLogger(__name__).warning("Alembic startup migration failed: %s", _mig_err)

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# FastAPI application + middleware
# ---------------------------------------------------------------------------

app = FastAPI()

# Rate-limit error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — origins from environment; default to nothing if unset
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Security headers + CSP
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "0"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    return response


# ---------------------------------------------------------------------------
# Desk router (Phase 7)
# ---------------------------------------------------------------------------

from app.web.desk import router as _desk_router  # noqa: E402
app.include_router(_desk_router)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class EmailRequest(BaseModel):
    text: str


class UpdateTaskRequest(BaseModel):
    title: str
    deadline: str
    priority: str


class UpdateTaskAIRequest(BaseModel):
    text: str


class DeleteTaskAIRequest(BaseModel):
    text: str


class AssistantRequest(BaseModel):
    text: str


class IngestRequest(BaseModel):
    """
    Payload for POST /ingest.

    Fields mirror NormalizedMessage so callers (workers, tests) can POST
    a message directly without going through a live mail provider.
    """
    provider: str
    provider_message_id: str
    message_id_header: Optional[str] = None
    subject: str
    sender: str
    recipients: list = []
    body_text: str = ""
    attachments: list = []   # list of {filename, content_type, size_bytes}
    raw: dict = {}


class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def normalize_email(email):
    return str(email).strip().lower()


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    """Return an Argon2id hash of the password."""
    return _ph.hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    """
    Verify password against stored hash.
    Supports both Argon2id hashes and legacy SHA-256 hashes (one-time upgrade).
    Returns True if password is correct.
    Raises VerifyMismatchError if wrong.
    """
    if stored_hash.startswith("$argon2"):
        _ph.verify(stored_hash, password)
        return True
    # Legacy SHA-256 path
    if stored_hash == _sha256_hex(password):
        return True
    raise VerifyMismatchError("password mismatch")


def upgrade_password_if_needed(db, user: "User", password: str):
    """Re-hash a legacy SHA-256 hash to Argon2id in-place (one-time)."""
    if not user.password_hash.startswith("$argon2"):
        user.password_hash = hash_password(password)
        db.commit()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(db, user_id: int, tenant_id: int) -> str:
    """Create a new session; rotate (delete all old sessions first)."""
    db.query(UserSession).filter(UserSession.user_id == user_id).delete()
    raw_token = secrets.token_hex(32)
    session = UserSession(
        user_id=user_id,
        tenant_id=tenant_id,
        token_hash=_token_hash(raw_token),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS),
    )
    db.add(session)
    db.commit()
    return raw_token


def get_session_user(db, raw_token: str) -> "Optional[User]":
    """Look up a valid (non-expired) session and return its User."""
    th = _token_hash(raw_token)
    session = db.query(UserSession).filter(UserSession.token_hash == th).first()
    if not session:
        return None
    if session.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        db.delete(session)
        db.commit()
        return None
    return db.query(User).filter(User.id == session.user_id).first()


def get_current_user_from_token(db, authorization: Optional[str]) -> "Optional[User]":
    if not authorization or not authorization.startswith("Bearer "):
        return None
    raw_token = authorization.removeprefix("Bearer ").strip()
    if not raw_token:
        return None
    return get_session_user(db, raw_token)


def require_current_user(db, authorization: Optional[str]) -> "User":
    user = get_current_user_from_token(db, authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def safe_db_error_message(error: Exception) -> str:
    if isinstance(error, OperationalError):
        return "Database connection failed. Please try again."
    return "Internal server error"


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------

def serialize_user(user: User) -> dict:
    return {"id": user.id, "name": user.name, "email": user.email}


def serialize_task(task: Task) -> dict:
    return {
        "id": task.id,
        "title": task.title,
        "deadline": task.deadline,
        "priority": task.priority,
        "status": task.status or "active",
        "user_id": task.user_id,
    }


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------

def normalize_text_for_match(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def get_active_tasks_query(db, user_id: int, tenant_id: int):
    return (
        db.query(Task)
        .filter(Task.tenant_id == tenant_id)
        .filter(Task.user_id == user_id)
        .filter(or_(Task.status.is_(None), Task.status != "completed"))
    )


def get_active_tasks(db, user_id: int, tenant_id: int):
    return get_active_tasks_query(db, user_id, tenant_id).order_by(Task.id.desc()).all()


def extract_delete_search_text(text: str) -> str:
    normalized = normalize_text_for_match(text)
    for prefix in ["delete ", "delete task ", "remove ", "remove task ",
                   "erase ", "drop ", "cancel "]:
        if normalized.startswith(prefix):
            candidate = normalized[len(prefix):].strip()
            if candidate:
                return candidate
    return normalized


def extract_complete_search_text(text: str) -> str:
    normalized = normalize_text_for_match(text)
    for prefix in ["complete ", "complete task ", "mark ", "mark task ",
                   "finish ", "done ", "انجام ", "انجامش کن ", "تمام ", "تموم "]:
        if normalized.startswith(prefix):
            candidate = normalized[len(prefix):].strip()
            if candidate:
                return candidate
    return normalized


def extract_update_search_text(text: str) -> str:
    normalized = normalize_text_for_match(text)
    for prefix in ["update ", "update task ", "edit ", "edit task ",
                   "change ", "change task ", "modify ", "modify task ", "set "]:
        if normalized.startswith(prefix):
            candidate = normalized[len(prefix):].strip()
            if candidate:
                return candidate
    return normalized


def is_show_tasks_request(text: str) -> bool:
    normalized = normalize_text_for_match(text)
    exact = {"show my tasks", "show all tasks", "show tasks", "list my tasks",
             "list all tasks", "list tasks", "display my tasks", "display all tasks",
             "display tasks", "my tasks", "all tasks"}
    if normalized in exact:
        return True
    for pattern in ["show my tasks", "show all tasks", "list my tasks",
                    "list all tasks", "display my tasks", "display all tasks"]:
        if pattern in normalized:
            return True
    return False


def detect_tasks_query_request(text: str):
    normalized = normalize_text_for_match(text)
    if not normalized:
        return None
    trigger_words = ["show", "list", "display"]
    has_task_word = "task" in normalized
    has_query_trigger = any(w in normalized for w in trigger_words)
    if not has_task_word and normalized not in {"my tasks", "all tasks"}:
        return None
    if not has_query_trigger and normalized not in {"my tasks", "all tasks"}:
        return None
    priority = None
    if "high priority" in normalized or normalized in {"high tasks", "high priority tasks"}:
        priority = "high"
    elif "medium priority" in normalized or normalized in {"medium tasks", "medium priority tasks"}:
        priority = "medium"
    elif "low priority" in normalized or normalized in {"low tasks", "low priority tasks"}:
        priority = "low"
    return {"priority": priority}


def detect_memory_reference(text: str) -> bool:
    normalized = normalize_text_for_match(text)
    if normalized in {"it", "that", "this", "انجام شد", "تموم شد", "تمام شد"}:
        return True
    for marker in ("it ", "that ", "this "):
        if normalized.startswith(marker):
            return True
    for marker in (" it", " that", " this"):
        if normalized.endswith(marker):
            return True
    padded = f" {normalized} "
    return " it " in padded or " that " in padded or " this " in padded


def detect_completion_request(text: str) -> bool:
    normalized = normalize_text_for_match(text)
    exact = {"done", "completed", "complete it", "mark it done", "mark it completed",
             "finish it", "انجام شد", "تموم شد", "تمام شد",
             "این انجام شد", "این تموم شد", "این تمام شد"}
    if normalized in exact:
        return True
    for pattern in ["complete it", "mark it done", "mark it completed", "finish it",
                    "done it", "completed it", "انجام شد", "تموم شد", "تمام شد"]:
        if pattern in normalized:
            return True
    return False


def build_tasks_list_message(tasks) -> str:
    if not tasks:
        return "You have no tasks."
    lines = ["Here are your tasks:"]
    for task in tasks:
        lines.append(f"#{task.id} - {task.title} | Deadline: {task.deadline} | Priority: {task.priority}")
    return "\n".join(lines)


def build_filtered_tasks_message(tasks, priority: Optional[str] = None) -> str:
    if not tasks:
        return f"You have no {priority} priority tasks right now." if priority else "You have no tasks."
    lines = [f"Here are your {priority} priority tasks:"] if priority else ["Here are your tasks:"]
    for task in tasks:
        lines.append(f"#{task.id} - {task.title} | Deadline: {task.deadline} | Priority: {task.priority}")
    return "\n".join(lines)


def get_user_last_task(db, user: User):
    if not user.last_task_id:
        return None
    return db.query(Task).filter(
        Task.id == user.last_task_id,
        Task.tenant_id == user.tenant_id,
        Task.user_id == user.id,
    ).first()


def remember_task(db, user: User, task: Optional[Task]):
    if not task:
        return
    user.last_task_id = task.id
    db.commit()
    db.refresh(user)


def clear_remembered_task(db, user: User, task: Optional[Task] = None):
    if task is not None and user.last_task_id != task.id:
        return
    if user.last_task_id is None:
        return
    user.last_task_id = None
    db.commit()
    db.refresh(user)


def mark_task_completed(db, user: User, task: Task):
    task.status = "completed"
    db.commit()
    db.refresh(task)
    clear_remembered_task(db, user, task)
    return task


def build_task_brief(task: Task) -> str:
    return f"#{task.id} - {task.title} | Deadline: {task.deadline} | Priority: {task.priority}"


def build_clarify_message_for_tasks(tasks, action_word: str) -> str:
    if not tasks:
        return f"Which task would you like me to {action_word}?"
    lines = [f"Which task would you like me to {action_word}?"]
    for task in tasks[:5]:
        lines.append(build_task_brief(task))
    return "\n".join(lines)


def get_task_by_id_from_list(tasks, task_id):
    try:
        normalized_id = int(task_id)
    except (TypeError, ValueError):
        return None
    for task in tasks:
        if task.id == normalized_id:
            return task
    return None


def score_task_match(task: Task, search_text: str) -> int:
    normalized_search = normalize_text_for_match(search_text)
    if not normalized_search:
        return 0
    search_words = [w for w in normalized_search.split(" ") if w]
    if not search_words:
        return 0
    title = normalize_text_for_match(task.title)
    deadline = normalize_text_for_match(task.deadline)
    priority = normalize_text_for_match(task.priority)
    status = normalize_text_for_match(task.status or "active")
    task_id_text = str(task.id)
    score = 0
    if normalized_search == title:
        score += 1000
    if normalized_search in title:
        score += 500
    if normalized_search == deadline:
        score += 250
    if normalized_search in deadline:
        score += 100
    if normalized_search == priority:
        score += 100
    if normalized_search in priority:
        score += 50
    if normalized_search == status:
        score += 50
    if normalized_search in (task_id_text, f"#{task_id_text}"):
        score += 1200
    matched_words = 0
    for word in search_words:
        if word in (task_id_text, f"#{task_id_text}"):
            score += 300
            matched_words += 1
        elif word in title:
            score += 40
            matched_words += 1
        elif word in deadline:
            score += 15
            matched_words += 1
        elif word in priority:
            score += 10
            matched_words += 1
        elif word in status:
            score += 5
            matched_words += 1
    if matched_words == len(search_words):
        score += 200
    return score


def find_matching_tasks(tasks, user_text: str):
    normalized_search = normalize_text_for_match(user_text)
    if not normalized_search:
        return []
    scored = []
    for task in tasks:
        s = score_task_match(task, normalized_search)
        if s > 0:
            scored.append((s, task.id, task))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored]


def find_best_task_match(tasks, user_text: str):
    matches = find_matching_tasks(tasks, extract_delete_search_text(user_text))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    top = score_task_match(matches[0], extract_delete_search_text(user_text))
    second = score_task_match(matches[1], extract_delete_search_text(user_text))
    return matches[0] if top != second else None


def resolve_task_reference(tasks, full_request_text: str, action_item: Optional[dict] = None,
                           remembered_task: Optional[Task] = None,
                           use_memory_reference: bool = False,
                           action_type: str = "update") -> dict:
    if use_memory_reference and remembered_task:
        return {"type": "resolved", "task": remembered_task}
    action_item = action_item or {}
    ai_task_id = action_item.get("task_id")
    if ai_task_id is not None:
        task_by_id = get_task_by_id_from_list(tasks, ai_task_id)
        if task_by_id:
            return {"type": "resolved", "task": task_by_id}
    candidate_texts = []
    if action_type == "delete":
        candidate_texts.append(extract_delete_search_text(full_request_text))
    elif action_type == "update":
        candidate_texts.append(extract_update_search_text(full_request_text))
    else:
        candidate_texts.append(normalize_text_for_match(full_request_text))
    if action_item.get("title"):
        candidate_texts.insert(0, action_item.get("title", ""))
    if action_item.get("deadline"):
        candidate_texts.append(action_item.get("deadline", ""))
    checked_texts: list = []
    best_matches: list = []
    for text_value in candidate_texts:
        normalized = normalize_text_for_match(text_value)
        if not normalized or normalized in checked_texts:
            continue
        checked_texts.append(normalized)
        matches = find_matching_tasks(tasks, normalized)
        if matches:
            best_matches = matches
            break
    if not best_matches:
        return {"type": "not_found"}
    if len(best_matches) == 1:
        return {"type": "resolved", "task": best_matches[0]}
    first_score = score_task_match(best_matches[0], checked_texts[0])
    second_score = score_task_match(best_matches[1], checked_texts[0])
    if first_score > second_score:
        return {"type": "resolved", "task": best_matches[0]}
    return {"type": "ambiguous", "tasks": best_matches[:5]}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return FileResponse("index.html")


@app.get("/health")
def health():
    """Liveness probe — always returns ok if the process is up."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness probe — checks the database."""
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception:
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    finally:
        db.close()


@app.post("/auth/signup")
@limiter.limit("5/minute")
def signup(request: Request, body: SignupRequest):
    db = SessionLocal()
    try:
        email = normalize_email(body.email)
        password = body.password.strip()
        name = body.name.strip()
        if db.query(User).filter(User.email == email).first():
            raise HTTPException(status_code=400, detail="Email already exists")
        if len(password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
        # Resolve or create the default tenant for self-service signup.
        tenant = db.query(Tenant).filter(Tenant.slug == "default").first()
        if not tenant:
            tenant = Tenant(name="Default", slug="default")
            db.add(tenant)
            db.commit()
            db.refresh(tenant)
        user = User(
            tenant_id=tenant.id,
            name=name,
            email=email,
            password_hash=hash_password(password),
            last_task_id=None,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return {"message": "User created successfully", "user": serialize_user(user)}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.post("/auth/login")
@limiter.limit("10/minute")
def login(request: Request, body: LoginRequest):
    db = SessionLocal()
    try:
        email = normalize_email(body.email)
        password = body.password.strip()
        user = db.query(User).filter(User.email == email).first()
        if not user or not user.password_hash:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        try:
            verify_password(user.password_hash, password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        # One-time upgrade: re-hash legacy SHA-256 to Argon2id
        upgrade_password_if_needed(db, user, password)
        # Rotate session (delete old, create new)
        raw_token = create_session(db, user.id, user.tenant_id)
        return {
            "message": "Login successful",
            "token": raw_token,
            "user": serialize_user(user),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.post("/auth/logout")
def logout(authorization: Optional[str] = Header(default=None)):
    db = SessionLocal()
    try:
        user = require_current_user(db, authorization)
        db.query(UserSession).filter(UserSession.user_id == user.id).delete()
        db.commit()
        return {"message": "Logged out successfully"}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.get("/auth/me")
def auth_me(authorization: Optional[str] = Header(default=None)):
    db = SessionLocal()
    try:
        user = require_current_user(db, authorization)
        return {"message": "Authenticated user", "user": serialize_user(user)}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.get("/stats")
def get_stats(authorization: Optional[str] = Header(default=None)):
    db = SessionLocal()
    try:
        user = require_current_user(db, authorization)
        tasks = get_active_tasks(db, user.id, user.tenant_id)
        high_count = sum(1 for t in tasks if (t.priority or "").lower() == "high")
        medium_count = sum(1 for t in tasks if (t.priority or "").lower() == "medium")
        low_count = sum(1 for t in tasks if (t.priority or "").lower() == "low")
        return {
            "total_tasks": len(tasks),
            "high_priority_tasks": high_count,
            "medium_priority_tasks": medium_count,
            "low_priority_tasks": low_count,
            "recent_tasks": [serialize_task(t) for t in tasks[:5]],
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.get("/tasks")
def get_tasks(authorization: Optional[str] = Header(default=None)):
    db = SessionLocal()
    try:
        user = require_current_user(db, authorization)
        tasks = get_active_tasks(db, user.id, user.tenant_id)
        result = [serialize_task(t) for t in tasks]
        return {"count": len(result), "tasks": result}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.get("/tasks/search")
def search_tasks(
    q: str = Query(default=""),
    priority: str = Query(default=""),
    deadline: str = Query(default=""),
    authorization: Optional[str] = Header(default=None),
):
    db = SessionLocal()
    try:
        user = require_current_user(db, authorization)
        query = get_active_tasks_query(db, user.id, user.tenant_id)
        if q.strip():
            query = query.filter(
                or_(
                    Task.title.ilike(f"%{q}%"),
                    Task.deadline.ilike(f"%{q}%"),
                    Task.priority.ilike(f"%{q}%"),
                )
            )
        if priority.strip():
            query = query.filter(Task.priority.ilike(priority.strip()))
        if deadline.strip():
            query = query.filter(Task.deadline.ilike(f"%{deadline.strip()}%"))
        tasks = query.order_by(Task.id.desc()).all()
        result = [serialize_task(t) for t in tasks]
        return {"count": len(result), "tasks": result}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.get("/tasks/{task_id}")
def get_task(task_id: int, authorization: Optional[str] = Header(default=None)):
    db = SessionLocal()
    try:
        user = require_current_user(db, authorization)
        task = db.query(Task).filter(Task.id == task_id, Task.tenant_id == user.tenant_id, Task.user_id == user.id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return serialize_task(task)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.delete("/tasks/{task_id}")
def delete_task(task_id: int, authorization: Optional[str] = Header(default=None)):
    db = SessionLocal()
    try:
        user = require_current_user(db, authorization)
        task = db.query(Task).filter(Task.id == task_id, Task.tenant_id == user.tenant_id, Task.user_id == user.id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        deleted_task = serialize_task(task)
        db.delete(task)
        db.commit()
        clear_remembered_task(db, user, task)
        return {"message": "Task deleted successfully", "deleted_task": deleted_task}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.put("/tasks/{task_id}")
def update_task(task_id: int, body: UpdateTaskRequest, authorization: Optional[str] = Header(default=None)):
    db = SessionLocal()
    try:
        user = require_current_user(db, authorization)
        task = db.query(Task).filter(Task.id == task_id, Task.tenant_id == user.tenant_id, Task.user_id == user.id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        task.title = body.title
        task.deadline = body.deadline
        task.priority = body.priority
        task.status = task.status or "active"
        db.commit()
        db.refresh(task)
        remember_task(db, user, task)
        return {"message": "Task updated successfully", "task": serialize_task(task)}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.post("/tasks/{task_id}/ai-update")
@limiter.limit("20/minute")
def ai_update_task(request: Request, task_id: int, body: UpdateTaskAIRequest,
                   authorization: Optional[str] = Header(default=None)):
    db = SessionLocal()
    try:
        user = require_current_user(db, authorization)
        task = db.query(Task).filter(Task.id == task_id, Task.tenant_id == user.tenant_id, Task.user_id == user.id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        prompt = f"""You are an AI office assistant.

You will update an existing task based on the user's instruction.
Return ONLY valid JSON. Do not include markdown, code fences, or extra text.

Current task:
{json.dumps(serialize_task(task))}

User instruction:
{body.text}

Rules:
- Keep the original value if the user did not ask to change it.
- priority must be one of: low, medium, high

Required JSON format:
{{
  "title": "updated task title",
  "deadline": "updated deadline or current deadline",
  "priority": "low, medium, or high"
}}"""
        response = client.responses.create(model="gpt-4.1-mini", input=prompt)
        output_text = response.output_text.strip()
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="AI response was not valid JSON")
        task.title = parsed.get("title", task.title) or task.title
        task.deadline = parsed.get("deadline", task.deadline) or task.deadline
        task.priority = parsed.get("priority", task.priority) or task.priority
        task.status = task.status or "active"
        db.commit()
        db.refresh(task)
        remember_task(db, user, task)
        return {"message": "Task updated successfully", "task": serialize_task(task)}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.post("/tasks/ai-delete")
def ai_delete_task(body: DeleteTaskAIRequest, authorization: Optional[str] = Header(default=None)):
    db = SessionLocal()
    try:
        user = require_current_user(db, authorization)
        tasks = get_active_tasks(db, user.id, user.tenant_id)
        if not tasks:
            raise HTTPException(status_code=404, detail="No tasks found")
        remembered_task = get_user_last_task(db, user)
        use_memory_reference = detect_memory_reference(body.text)
        resolved = resolve_task_reference(
            tasks=tasks,
            full_request_text=body.text,
            action_item={},
            remembered_task=remembered_task,
            use_memory_reference=use_memory_reference,
            action_type="delete",
        )
        if resolved["type"] == "not_found":
            raise HTTPException(status_code=404, detail="Task not found")
        if resolved["type"] == "ambiguous":
            return {
                "action": "clarify",
                "message": build_clarify_message_for_tasks(resolved["tasks"], "delete"),
                "tasks": [serialize_task(t) for t in resolved["tasks"]],
            }
        task = db.query(Task).filter(
            Task.id == resolved["task"].id, Task.tenant_id == user.tenant_id, Task.user_id == user.id
        ).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        deleted_task = serialize_task(task)
        db.delete(task)
        db.commit()
        clear_remembered_task(db, user, task)
        return {"message": "Task deleted successfully", "deleted_task": deleted_task}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.post("/assistant")
@limiter.limit("30/minute")
def assistant(request: Request, body: AssistantRequest,
              authorization: Optional[str] = Header(default=None)):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    db = SessionLocal()
    try:
        user = require_current_user(db, authorization)
        tasks = get_active_tasks(db, user.id, user.tenant_id)
        task_list = [serialize_task(t) for t in tasks]
        remembered_task = get_user_last_task(db, user)
        use_memory_reference = detect_memory_reference(body.text)

        if detect_completion_request(body.text):
            target_task = remembered_task
            if not target_task:
                target_task = find_best_task_match(tasks, extract_complete_search_text(body.text))
            if not target_task and len(tasks) == 1:
                target_task = tasks[0]
            if not target_task:
                return {"action": "clarify", "message": "Please tell me which task was completed."}
            completed_task = mark_task_completed(db, user, target_task)
            return {"action": "complete", "message": "Task marked as completed",
                    "task": serialize_task(completed_task)}

        tasks_query = detect_tasks_query_request(body.text)
        if tasks_query:
            filtered_tasks = tasks
            if tasks_query["priority"]:
                filtered_tasks = [
                    t for t in tasks
                    if normalize_text_for_match(t.priority) == tasks_query["priority"]
                ]
            if filtered_tasks:
                remember_task(db, user, filtered_tasks[0])
            return {
                "action": "list",
                "message": build_filtered_tasks_message(filtered_tasks, tasks_query["priority"]),
                "count": len(filtered_tasks),
                "tasks": [serialize_task(t) for t in filtered_tasks],
            }

        if is_show_tasks_request(body.text):
            if tasks:
                remember_task(db, user, tasks[0])
            return {
                "action": "list",
                "message": build_tasks_list_message(tasks),
                "count": len(task_list),
                "tasks": task_list,
            }

        memory_block = ""
        if remembered_task:
            memory_block = f"""
Last remembered task for this user:
{json.dumps(serialize_task(remembered_task))}

Memory rules:
- If the user says "it", "that", "this", "delete it", "update it", or similar, use the remembered task.
- When the user's wording clearly refers to the remembered task, return that task_id.
"""

        prompt = f"""You are an AI office assistant.

Your job is to detect one or more user intents and return ONLY valid JSON.
Do not include markdown, code fences, or extra text.

Available actions: create, update, delete, clarify

Current active task list:
{json.dumps(task_list)}

{memory_block}

User instruction:
{body.text}

Rules:
- Break the instruction into one or more actions when needed.
- Use clarify if ambiguous or missing information.
- Use create only for a brand new task.
- Use update only when the user clearly refers to one existing task and wants it changed.
- Use delete only when the user clearly refers to one existing task and wants it removed.
- For update and delete, choose exactly one best matching task_id when possible.
- If multiple tasks could match, use clarify.
- For create: task_id null, return title, deadline, priority.
- For update: return task_id and updated title, deadline, priority.
- For delete: return task_id; title, deadline, priority can be empty strings.
- For clarify: task_id null, return a short question in clarify_message.
- priority must be one of: low, medium, high
- Always return a JSON object with an "actions" array.

Required JSON format:
{{
  "actions": [
    {{
      "action": "create or update or delete or clarify",
      "task_id": 123,
      "title": "task title or empty string",
      "deadline": "deadline or Not specified or empty string",
      "priority": "low or medium or high or empty string",
      "clarify_message": "question or empty string"
    }}
  ]
}}"""

        response = client.responses.create(model="gpt-4.1-mini", input=prompt)
        output_text = response.output_text.strip()
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="AI response was not valid JSON")

        actions = parsed if isinstance(parsed, list) else parsed.get("actions", [])
        if not actions:
            raise HTTPException(status_code=400, detail="No actions returned by AI")

        if use_memory_reference and remembered_task:
            for item in actions:
                if item.get("action") in {"update", "delete"} and not item.get("task_id"):
                    item["task_id"] = remembered_task.id

        for item in actions:
            if item.get("action") == "clarify":
                return {
                    "action": "clarify",
                    "message": item.get("clarify_message", "Please clarify your request."),
                    "actions": actions,
                }

        results = []
        for item in actions:
            action = item.get("action")
            current_tasks = get_active_tasks(db, user.id, user.tenant_id)

            if action == "create":
                db_task = Task(
                    tenant_id=user.tenant_id,
                    user_id=user.id,
                    title=item.get("title", ""),
                    deadline=item.get("deadline", "Not specified"),
                    priority=item.get("priority", "medium"),
                    status="active",
                )
                db.add(db_task)
                db.commit()
                db.refresh(db_task)
                remember_task(db, user, db_task)
                results.append({"action": "create", "message": "Task created successfully",
                                 "task": serialize_task(db_task)})

            elif action == "update":
                resolved = resolve_task_reference(
                    tasks=current_tasks, full_request_text=body.text,
                    action_item=item, remembered_task=remembered_task,
                    use_memory_reference=use_memory_reference, action_type="update",
                )
                if resolved["type"] == "not_found":
                    return {"action": "clarify", "message": "Which task would you like me to update?"}
                if resolved["type"] == "ambiguous":
                    return {
                        "action": "clarify",
                        "message": build_clarify_message_for_tasks(resolved["tasks"], "update"),
                        "tasks": [serialize_task(t) for t in resolved["tasks"]],
                    }
                task = db.query(Task).filter(
                    Task.id == resolved["task"].id, Task.tenant_id == user.tenant_id, Task.user_id == user.id
                ).first()
                if not task:
                    raise HTTPException(status_code=404, detail="Task not found")
                if item.get("title"):
                    task.title = item["title"]
                if item.get("deadline"):
                    task.deadline = item["deadline"]
                if item.get("priority"):
                    task.priority = item["priority"]
                task.status = task.status or "active"
                db.commit()
                db.refresh(task)
                remember_task(db, user, task)
                results.append({"action": "update", "message": "Task updated successfully",
                                 "task": serialize_task(task)})

            elif action == "delete":
                resolved = resolve_task_reference(
                    tasks=current_tasks, full_request_text=body.text,
                    action_item=item, remembered_task=remembered_task,
                    use_memory_reference=use_memory_reference, action_type="delete",
                )
                if resolved["type"] == "not_found":
                    return {"action": "clarify", "message": "Which task would you like me to delete?"}
                if resolved["type"] == "ambiguous":
                    return {
                        "action": "clarify",
                        "message": build_clarify_message_for_tasks(resolved["tasks"], "delete"),
                        "tasks": [serialize_task(t) for t in resolved["tasks"]],
                    }
                task = db.query(Task).filter(
                    Task.id == resolved["task"].id, Task.tenant_id == user.tenant_id, Task.user_id == user.id
                ).first()
                if not task:
                    raise HTTPException(status_code=404, detail="Task not found")
                deleted_task = serialize_task(task)
                db.delete(task)
                db.commit()
                clear_remembered_task(db, user, task)
                results.append({"action": "delete", "message": "Task deleted successfully",
                                 "task": deleted_task})

            else:
                raise HTTPException(status_code=400, detail="Invalid action returned by AI")

        if len(results) == 1:
            return results[0]
        return {"message": "Multiple actions completed successfully", "results": results}

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.post("/analyze")
@limiter.limit("20/minute")
def analyze(request: Request, body: EmailRequest,
            authorization: Optional[str] = Header(default=None)):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    # Authentication required before any model call (row 35 model-route gate)
    db = SessionLocal()
    try:
        user = require_current_user(db, authorization)

        prompt = f"""You are an AI office assistant.

Analyze the email below and return ONLY valid JSON.
Do not include markdown, code fences, or extra text.

Required JSON format:
{{
  "summary": "short summary",
  "tasks": [
    {{
      "title": "task title",
      "deadline": "deadline or Not specified",
      "priority": "low, medium, or high"
    }}
  ]
}}

Email:
{body.text}"""

        response = client.responses.create(model="gpt-4.1-mini", input=prompt)
        output_text = response.output_text.strip()
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="AI response was not valid JSON")

        saved_tasks = []
        for task in parsed.get("tasks", []):
            db_task = Task(
                tenant_id=user.tenant_id,
                user_id=user.id,
                title=task.get("title", ""),
                deadline=task.get("deadline", "Not specified"),
                priority=task.get("priority", "medium"),
                status="active",
            )
            db.add(db_task)
            db.commit()
            db.refresh(db_task)
            remember_task(db, user, db_task)
            saved_tasks.append(serialize_task(db_task))

        return {
            "original_text": body.text,
            "summary": parsed.get("summary", ""),
            "created_tasks": saved_tasks,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Ingest route (Phase 4)
# ---------------------------------------------------------------------------

@app.post("/ingest")
def ingest(body: IngestRequest, authorization: Optional[str] = Header(default=None)):
    """
    Accept a normalised message payload, run duplicate/idempotency/quarantine
    checks, and store the raw message.  Returns the stored record and result.

    Authentication required.  The authenticated user's tenant_id is used as
    the ingest tenant.
    """
    from app.ingest.ingest import ingest_message
    from app.ingest.normalize import Attachment, NormalizedMessage

    db = SessionLocal()
    try:
        user = require_current_user(db, authorization)

        attachments = [
            Attachment(
                filename=a.get("filename", ""),
                content_type=a.get("content_type", "application/octet-stream"),
                size_bytes=int(a.get("size_bytes", 0)),
            )
            for a in (body.attachments or [])
        ]

        msg = NormalizedMessage(
            provider=body.provider,
            provider_message_id=body.provider_message_id,
            tenant_id=user.tenant_id,
            message_id_header=body.message_id_header,
            subject=body.subject,
            subject_normalized=NormalizedMessage.normalize_subject(body.subject),
            sender=body.sender,
            recipients=body.recipients,
            body_text=body.body_text,
            attachments=attachments,
            raw=body.raw or {},
        )

        record, result = ingest_message(db, msg)

        return {
            "result": result,
            "message_id": record.id,
            "state": record.state,
            "attachment_state": record.attachment_state,
            "ingest_time": record.ingest_time.isoformat(),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()
