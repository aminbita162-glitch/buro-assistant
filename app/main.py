import json
import os
import hashlib
import secrets
import re

from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from openai import OpenAI

from sqlalchemy import create_engine, Column, Integer, String, or_, text, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_timeout=30
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    token = Column(String, nullable=True, index=True)
    last_task_id = Column(Integer, nullable=True)


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    deadline = Column(String)
    priority = Column(String)
    status = Column(String, nullable=True)
    user_id = Column(Integer, index=True, nullable=True)


Base.metadata.create_all(bind=engine)


def ensure_tasks_user_id_column():
    inspector = inspect(engine)
    columns = [column["name"] for column in inspector.get_columns("tasks")]

    if "user_id" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE tasks ADD COLUMN user_id INTEGER"))


def ensure_tasks_status_column():
    inspector = inspect(engine)
    columns = [column["name"] for column in inspector.get_columns("tasks")]

    if "status" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE tasks ADD COLUMN status VARCHAR"))


def ensure_users_token_column():
    inspector = inspect(engine)
    columns = [column["name"] for column in inspector.get_columns("users")]

    if "token" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE users ADD COLUMN token VARCHAR"))


def ensure_users_last_task_id_column():
    inspector = inspect(engine)
    columns = [column["name"] for column in inspector.get_columns("users")]

    if "last_task_id" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE users ADD COLUMN last_task_id INTEGER"))


ensure_tasks_user_id_column()
ensure_tasks_status_column()
ensure_users_token_column()
ensure_users_last_task_id_column()


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


class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


def normalize_email(email):
    return str(email).strip().lower()


def hash_password(password: str):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def serialize_user(user: User):
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
    }


def serialize_task(task: Task):
    return {
        "id": task.id,
        "title": task.title,
        "deadline": task.deadline,
        "priority": task.priority,
        "status": task.status or "active",
        "user_id": task.user_id,
    }


def get_current_user_from_token(db, authorization: str | None):
    if not authorization:
        return None

    if not authorization.startswith("Bearer "):
        return None

    token = authorization.replace("Bearer ", "").strip()

    if not token:
        return None

    user = db.query(User).filter(User.token == token).first()
    return user


def require_current_user(db, authorization: str | None):
    user = get_current_user_from_token(db, authorization)

    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return user


def safe_db_error_message(error: Exception):
    if isinstance(error, OperationalError):
        return "Database connection failed. Please try again."
    return "Internal server error"


def normalize_text_for_match(value: str):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def get_active_tasks_query(db, user_id: int):
    return (
        db.query(Task)
        .filter(Task.user_id == user_id)
        .filter(or_(Task.status.is_(None), Task.status != "completed"))
    )


def get_active_tasks(db, user_id: int):
    return get_active_tasks_query(db, user_id).order_by(Task.id.desc()).all()


def extract_delete_search_text(text: str):
    normalized = normalize_text_for_match(text)
    prefixes = [
        "delete ",
        "remove ",
        "erase ",
        "drop ",
        "cancel "
    ]

    for prefix in prefixes:
        if normalized.startswith(prefix):
            candidate = normalized[len(prefix):].strip()
            if candidate:
                return candidate

    return normalized


def extract_complete_search_text(text: str):
    normalized = normalize_text_for_match(text)
    prefixes = [
        "complete ",
        "complete task ",
        "mark ",
        "mark task ",
        "finish ",
        "done ",
        "انجام ",
        "انجامش کن ",
        "تمام ",
        "تموم "
    ]

    for prefix in prefixes:
        if normalized.startswith(prefix):
            candidate = normalized[len(prefix):].strip()
            if candidate:
                return candidate

    return normalized


def find_best_task_match(tasks, user_text: str):
    search_text = extract_delete_search_text(user_text)

    if not search_text:
        return None

    normalized_search = normalize_text_for_match(search_text)
    search_words = [word for word in normalized_search.split(" ") if word]

    if not search_words:
        return None

    scored = []

    for task in tasks:
        title = normalize_text_for_match(task.title)
        deadline = normalize_text_for_match(task.deadline)
        priority = normalize_text_for_match(task.priority)
        status = normalize_text_for_match(task.status or "active")

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

        matched_words = 0
        for word in search_words:
            if word in title:
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

        if score > 0:
            scored.append((score, task.id, task))

    if not scored:
        return None

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top_score = scored[0][0]
    top_items = [item for item in scored if item[0] == top_score]

    if len(top_items) > 1:
        return None

    return scored[0][2]


def is_show_tasks_request(text: str):
    normalized = normalize_text_for_match(text)

    exact_matches = {
        "show my tasks",
        "show all tasks",
        "show tasks",
        "list my tasks",
        "list all tasks",
        "list tasks",
        "display my tasks",
        "display all tasks",
        "display tasks",
        "my tasks",
        "all tasks"
    }

    if normalized in exact_matches:
        return True

    patterns = [
        "show my tasks",
        "show all tasks",
        "list my tasks",
        "list all tasks",
        "display my tasks",
        "display all tasks"
    ]

    for pattern in patterns:
        if pattern in normalized:
            return True

    return False


def detect_tasks_query_request(text: str):
    normalized = normalize_text_for_match(text)

    if not normalized:
        return None

    trigger_words = ["show", "list", "display"]
    has_task_word = "task" in normalized
    has_query_trigger = any(word in normalized for word in trigger_words)

    if not has_task_word and normalized not in {"my tasks", "all tasks"}:
        return None

    if not has_query_trigger and normalized not in {"my tasks", "all tasks"}:
        return None

    priority = None

    if "high priority" in normalized or normalized == "high tasks" or normalized == "high priority tasks":
        priority = "high"
    elif "medium priority" in normalized or normalized == "medium tasks" or normalized == "medium priority tasks":
        priority = "medium"
    elif "low priority" in normalized or normalized == "low tasks" or normalized == "low priority tasks":
        priority = "low"

    return {
        "priority": priority
    }


def detect_memory_reference(text: str):
    normalized = normalize_text_for_match(text)

    memory_only_values = {
        "it",
        "that",
        "this",
        "انجام شد",
        "تموم شد",
        "تمام شد"
    }

    if normalized in memory_only_values:
        return True

    if normalized.startswith("it ") or normalized.startswith("that ") or normalized.startswith("this "):
        return True

    if normalized.endswith(" it") or normalized.endswith(" that") or normalized.endswith(" this"):
        return True

    if " it " in f" {normalized} ":
        return True

    if " that " in f" {normalized} ":
        return True

    if " this " in f" {normalized} ":
        return True

    return False


def detect_completion_request(text: str):
    normalized = normalize_text_for_match(text)

    exact_matches = {
        "done",
        "completed",
        "complete it",
        "mark it done",
        "mark it completed",
        "finish it",
        "انجام شد",
        "تموم شد",
        "تمام شد",
        "این انجام شد",
        "این تموم شد",
        "این تمام شد"
    }

    if normalized in exact_matches:
        return True

    patterns = [
        "complete it",
        "mark it done",
        "mark it completed",
        "finish it",
        "done it",
        "completed it",
        "انجام شد",
        "تموم شد",
        "تمام شد"
    ]

    for pattern in patterns:
        if pattern in normalized:
            return True

    return False


def build_tasks_list_message(tasks):
    if not tasks:
        return "You have no tasks."

    lines = ["Here are your tasks:"]
    for task in tasks:
        lines.append(
            f"#{task.id} - {task.title} | Deadline: {task.deadline} | Priority: {task.priority}"
        )

    return "\n".join(lines)


def build_filtered_tasks_message(tasks, priority: str | None = None):
    if not tasks:
        if priority:
            return f"You have no {priority} priority tasks right now."
        return "You have no tasks."

    if priority:
        lines = [f"Here are your {priority} priority tasks:"]
    else:
        lines = ["Here are your tasks:"]

    for task in tasks:
        lines.append(
            f"#{task.id} - {task.title} | Deadline: {task.deadline} | Priority: {task.priority}"
        )

    return "\n".join(lines)


def get_user_last_task(db, user: User):
    if not user.last_task_id:
        return None

    task = db.query(Task).filter(
        Task.id == user.last_task_id,
        Task.user_id == user.id
    ).first()

    return task


def remember_task(db, user: User, task: Task | None):
    if not task:
        return

    user.last_task_id = task.id
    db.commit()
    db.refresh(user)


def clear_remembered_task(db, user: User, task: Task | None = None):
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


@app.get("/")
def root():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/signup")
def signup(request: SignupRequest):
    db = SessionLocal()

    try:
        email = normalize_email(request.email)
        password = request.password.strip()
        name = request.name.strip()

        existing_user = db.query(User).filter(User.email == email).first()

        if existing_user:
            raise HTTPException(status_code=400, detail="Email already exists")

        if len(password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

        user = User(
            name=name,
            email=email,
            password_hash=hash_password(password),
            token=None,
            last_task_id=None
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        return {
            "message": "User created successfully",
            "user": serialize_user(user)
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.post("/auth/login")
def login(request: LoginRequest):
    db = SessionLocal()

    try:
        email = normalize_email(request.email)
        password = request.password.strip()
        password_hash = hash_password(password)

        user = db.query(User).filter(User.email == email).first()

        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        if not user.password_hash:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        if user.password_hash != password_hash:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        token = secrets.token_hex(24)
        user.token = token
        db.commit()
        db.refresh(user)

        return {
            "message": "Login successful",
            "token": token,
            "user": serialize_user(user)
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.post("/auth/logout")
def logout(authorization: str | None = Header(default=None)):
    db = SessionLocal()

    try:
        user = require_current_user(db, authorization)
        user.token = None
        db.commit()

        return {"message": "Logged out successfully"}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.get("/auth/me")
def auth_me(authorization: str | None = Header(default=None)):
    db = SessionLocal()

    try:
        user = require_current_user(db, authorization)

        return {
            "message": "Authenticated user",
            "user": serialize_user(user)
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.get("/stats")
def get_stats(authorization: str | None = Header(default=None)):
    db = SessionLocal()

    try:
        user = require_current_user(db, authorization)
        tasks = get_active_tasks(db, user.id)

        total = len(tasks)
        high_count = 0
        medium_count = 0
        low_count = 0

        for task in tasks:
            priority = (task.priority or "").lower()

            if priority == "high":
                high_count += 1
            elif priority == "medium":
                medium_count += 1
            elif priority == "low":
                low_count += 1

        recent_tasks = [serialize_task(task) for task in tasks[:5]]

        return {
            "total_tasks": total,
            "high_priority_tasks": high_count,
            "medium_priority_tasks": medium_count,
            "low_priority_tasks": low_count,
            "recent_tasks": recent_tasks
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.get("/tasks")
def get_tasks(authorization: str | None = Header(default=None)):
    db = SessionLocal()

    try:
        user = require_current_user(db, authorization)
        tasks = get_active_tasks(db, user.id)
        result = [serialize_task(task) for task in tasks]

        return {
            "count": len(result),
            "tasks": result
        }
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
    authorization: str | None = Header(default=None)
):
    db = SessionLocal()

    try:
        user = require_current_user(db, authorization)
        query = get_active_tasks_query(db, user.id)

        if q.strip():
            query = query.filter(
                or_(
                    Task.title.ilike(f"%{q}%"),
                    Task.deadline.ilike(f"%{q}%"),
                    Task.priority.ilike(f"%{q}%")
                )
            )

        if priority.strip():
            query = query.filter(Task.priority.ilike(priority.strip()))

        if deadline.strip():
            query = query.filter(Task.deadline.ilike(f"%{deadline.strip()}%"))

        tasks = query.order_by(Task.id.desc()).all()
        result = [serialize_task(task) for task in tasks]

        return {
            "count": len(result),
            "tasks": result
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.get("/tasks/{task_id}")
def get_task(task_id: int, authorization: str | None = Header(default=None)):
    db = SessionLocal()

    try:
        user = require_current_user(db, authorization)
        task = db.query(Task).filter(Task.id == task_id, Task.user_id == user.id).first()

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
def delete_task(task_id: int, authorization: str | None = Header(default=None)):
    db = SessionLocal()

    try:
        user = require_current_user(db, authorization)
        task = db.query(Task).filter(Task.id == task_id, Task.user_id == user.id).first()

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        db.delete(task)
        db.commit()
        clear_remembered_task(db, user, task)

        return {"message": "Task deleted successfully"}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.put("/tasks/{task_id}")
def update_task(task_id: int, request: UpdateTaskRequest, authorization: str | None = Header(default=None)):
    db = SessionLocal()

    try:
        user = require_current_user(db, authorization)
        task = db.query(Task).filter(Task.id == task_id, Task.user_id == user.id).first()

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        task.title = request.title
        task.deadline = request.deadline
        task.priority = request.priority
        task.status = task.status or "active"

        db.commit()
        db.refresh(task)
        remember_task(db, user, task)

        return {
            "message": "Task updated successfully",
            "task": serialize_task(task)
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.post("/tasks/{task_id}/ai-update")
def ai_update_task(task_id: int, request: UpdateTaskAIRequest, authorization: str | None = Header(default=None)):
    db = SessionLocal()

    try:
        user = require_current_user(db, authorization)
        task = db.query(Task).filter(Task.id == task_id, Task.user_id == user.id).first()

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        prompt = f"""
You are an AI office assistant.

You will update an existing task based on the user's instruction.
Return ONLY valid JSON.
Do not include markdown, code fences, or extra text.

Current task:
{json.dumps(serialize_task(task))}

User instruction:
{request.text}

Required JSON format:
{{
  "title": "updated task title",
  "deadline": "updated deadline or Not specified",
  "priority": "low, medium, or high"
}}
"""

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        output_text = response.output_text.strip()

        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail="AI response was not valid JSON"
            )

        task.title = parsed.get("title", task.title)
        task.deadline = parsed.get("deadline", task.deadline)
        task.priority = parsed.get("priority", task.priority)
        task.status = task.status or "active"

        db.commit()
        db.refresh(task)
        remember_task(db, user, task)

        return {
            "message": "Task updated successfully",
            "task": serialize_task(task)
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.post("/tasks/ai-delete")
def ai_delete_task(request: DeleteTaskAIRequest, authorization: str | None = Header(default=None)):
    db = SessionLocal()

    try:
        user = require_current_user(db, authorization)
        tasks = get_active_tasks(db, user.id)

        if not tasks:
            raise HTTPException(status_code=404, detail="No tasks found")

        fallback_task = None

        if detect_memory_reference(request.text):
            fallback_task = get_user_last_task(db, user)

        if not fallback_task:
            fallback_task = find_best_task_match(tasks, request.text)

        if fallback_task:
            task = db.query(Task).filter(Task.id == fallback_task.id, Task.user_id == user.id).first()

            if not task:
                raise HTTPException(status_code=404, detail="Task not found")

            deleted_task = serialize_task(task)

            db.delete(task)
            db.commit()
            clear_remembered_task(db, user, task)

            return {
                "message": "Task deleted successfully",
                "deleted_task": deleted_task
            }

        raise HTTPException(status_code=404, detail="Task not found")
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.post("/assistant")
def assistant(request: AssistantRequest, authorization: str | None = Header(default=None)):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    db = SessionLocal()

    try:
        user = require_current_user(db, authorization)
        tasks = get_active_tasks(db, user.id)
        task_list = [serialize_task(task) for task in tasks]
        remembered_task = get_user_last_task(db, user)
        use_memory_reference = detect_memory_reference(request.text)

        if detect_completion_request(request.text):
            target_task = None

            if remembered_task:
                target_task = remembered_task

            if not target_task:
                target_task = find_best_task_match(tasks, extract_complete_search_text(request.text))

            if not target_task and len(tasks) == 1:
                target_task = tasks[0]

            if not target_task:
                return {
                    "action": "clarify",
                    "message": "Please tell me which task was completed."
                }

            completed_task = mark_task_completed(db, user, target_task)

            return {
                "action": "complete",
                "message": "Task marked as completed",
                "task": serialize_task(completed_task)
            }

        tasks_query = detect_tasks_query_request(request.text)

        if tasks_query:
            filtered_tasks = tasks

            if tasks_query["priority"]:
                filtered_tasks = [
                    task for task in tasks
                    if normalize_text_for_match(task.priority) == tasks_query["priority"]
                ]

            if filtered_tasks:
                remember_task(db, user, filtered_tasks[0])

            return {
                "action": "list",
                "message": build_filtered_tasks_message(
                    filtered_tasks,
                    tasks_query["priority"]
                ),
                "count": len(filtered_tasks),
                "tasks": [serialize_task(task) for task in filtered_tasks]
            }

        if is_show_tasks_request(request.text):
            if tasks:
                remember_task(db, user, tasks[0])

            return {
                "action": "list",
                "message": build_tasks_list_message(tasks),
                "count": len(task_list),
                "tasks": task_list
            }

        memory_instruction_block = ""

        if remembered_task:
            memory_instruction_block = f"""
Last remembered task for this user:
{json.dumps(serialize_task(remembered_task))}

Memory rules:
- If the user says "it", "that", "this", "delete it", "update it", "make it tomorrow", or similar wording, use the remembered task.
- When the user's wording clearly refers to the remembered task, return that task_id.
- Do not ask for clarification if the remembered task makes the request clear enough.
"""

        prompt = f"""
You are an AI office assistant.

Your job is to detect one or more user intents and return ONLY valid JSON.
Do not include markdown, code fences, or extra text.

Available actions:
- create
- update
- delete
- clarify

Current active task list:
{json.dumps(task_list)}

{memory_instruction_block}

User instruction:
{request.text}

Rules:
- Break the user's instruction into one or more actions when needed.
- Use clarify if the instruction is ambiguous, unclear, or missing enough information.
- Use create only when the user wants a brand new task.
- Use update only when the user clearly refers to one existing task and wants it changed.
- Use delete only when the user clearly refers to one existing task and wants it removed.
- For update and delete, choose exactly one best matching task_id.
- Prefer exact title match first.
- If exact match does not exist, allow partial title match.
- If the user refers to only one distinctive word from the title such as "milk", use the matching task when there is only one clear best match.
- Only use clarify for delete when multiple tasks could match or no clear best match exists.
- Do not update multiple similar tasks at once.
- For create, set task_id to null and return title, deadline, and priority.
- For update, return the correct task_id and the updated title, deadline, and priority.
- For delete, return the correct task_id. Title, deadline, and priority can be empty strings.
- For clarify, set task_id to null and return a short question in clarify_message.
- Match against the current task list carefully.
- priority must be one of: low, medium, high
- Always return a JSON object with an "actions" array.
- If there is only one action, still return it inside the actions array.

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
}}
"""

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        output_text = response.output_text.strip()

        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail="AI response was not valid JSON"
            )

        if isinstance(parsed, list):
            actions = parsed
        else:
            actions = parsed.get("actions", [])

        if not actions:
            raise HTTPException(status_code=400, detail="No actions returned by AI")

        if use_memory_reference and remembered_task:
            for item in actions:
                if item.get("action") in {"update", "delete"}:
                    item["task_id"] = remembered_task.id

        for item in actions:
            if item.get("action") == "clarify":
                return {
                    "action": "clarify",
                    "message": item.get("clarify_message", "Please clarify your request."),
                    "actions": actions
                }

        results = []

        for item in actions:
            action = item.get("action")

            if action == "create":
                db_task = Task(
                    title=item.get("title", ""),
                    deadline=item.get("deadline", "Not specified"),
                    priority=item.get("priority", "medium"),
                    status="active",
                    user_id=user.id
                )
                db.add(db_task)
                db.commit()
                db.refresh(db_task)
                remember_task(db, user, db_task)

                results.append(
                    {
                        "action": "create",
                        "message": "Task created successfully",
                        "task": serialize_task(db_task)
                    }
                )

            elif action == "update":
                task_id = item.get("task_id")

                if use_memory_reference and remembered_task:
                    task_id = remembered_task.id

                task = db.query(Task).filter(Task.id == task_id, Task.user_id == user.id).first()

                if not task:
                    raise HTTPException(status_code=404, detail="Task not found")

                task.title = item.get("title", task.title)
                task.deadline = item.get("deadline", task.deadline)
                task.priority = item.get("priority", task.priority)
                task.status = task.status or "active"

                db.commit()
                db.refresh(task)
                remember_task(db, user, task)

                results.append(
                    {
                        "action": "update",
                        "message": "Task updated successfully",
                        "task": serialize_task(task)
                    }
                )

            elif action == "delete":
                fallback_task = None

                if use_memory_reference and remembered_task:
                    fallback_task = remembered_task

                if not fallback_task:
                    fallback_task = find_best_task_match(tasks, request.text)

                if not fallback_task:
                    raise HTTPException(status_code=400, detail="Please specify which task you want to delete")

                task = db.query(Task).filter(Task.id == fallback_task.id, Task.user_id == user.id).first()

                if not task:
                    raise HTTPException(status_code=404, detail="Task not found")

                deleted_task = serialize_task(task)

                db.delete(task)
                db.commit()
                clear_remembered_task(db, user, task)

                results.append(
                    {
                        "action": "delete",
                        "message": "Task deleted successfully",
                        "task": deleted_task
                    }
                )

            else:
                raise HTTPException(status_code=400, detail="Invalid action returned by AI")

        if len(results) == 1:
            return results[0]

        return {
            "message": "Multiple actions completed successfully",
            "results": results
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()


@app.post("/analyze")
def analyze(request: EmailRequest, authorization: str | None = Header(default=None)):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    prompt = f"""
You are an AI office assistant.

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
{request.text}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    output_text = response.output_text.strip()

    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="AI response was not valid JSON"
        )

    db = SessionLocal()

    try:
        user = require_current_user(db, authorization)
        saved_tasks = []

        for task in parsed.get("tasks", []):
            db_task = Task(
                title=task.get("title", ""),
                deadline=task.get("deadline", "Not specified"),
                priority=task.get("priority", "medium"),
                status="active",
                user_id=user.id
            )
            db.add(db_task)
            db.commit()
            db.refresh(db_task)
            remember_task(db, user, db_task)

            saved_tasks.append(serialize_task(db_task))

        return {
            "original_text": request.text,
            "summary": parsed.get("summary", ""),
            "created_tasks": saved_tasks
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=safe_db_error_message(error))
    finally:
        db.close()