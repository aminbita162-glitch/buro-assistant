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


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    deadline = Column(String)
    priority = Column(String)
    user_id = Column(Integer, index=True, nullable=True)


Base.metadata.create_all(bind=engine)


def ensure_tasks_user_id_column():
    inspector = inspect(engine)
    columns = [column["name"] for column in inspector.get_columns("tasks")]

    if "user_id" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE tasks ADD COLUMN user_id INTEGER"))


def ensure_users_token_column():
    inspector = inspect(engine)
    columns = [column["name"] for column in inspector.get_columns("users")]

    if "token" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE users ADD COLUMN token VARCHAR"))


ensure_tasks_user_id_column()
ensure_users_token_column()


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
            token=None
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
        tasks = db.query(Task).filter(Task.user_id == user.id).order_by(Task.id.desc()).all()

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
        tasks = db.query(Task).filter(Task.user_id == user.id).order_by(Task.id.desc()).all()
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
        query = db.query(Task).filter(Task.user_id == user.id)

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

        db.commit()
        db.refresh(task)

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

        db.commit()
        db.refresh(task)

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
        tasks = db.query(Task).filter(Task.user_id == user.id).order_by(Task.id.desc()).all()

        if not tasks:
            raise HTTPException(status_code=404, detail="No tasks found")

        fallback_task = find_best_task_match(tasks, request.text)

        if fallback_task:
            task = db.query(Task).filter(Task.id == fallback_task.id, Task.user_id == user.id).first()

            if not task:
                raise HTTPException(status_code=404, detail="Task not found")

            deleted_task = serialize_task(task)

            db.delete(task)
            db.commit()

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
        tasks = db.query(Task).filter(Task.user_id == user.id).order_by(Task.id.desc()).all()
        task_list = [serialize_task(task) for task in tasks]

        prompt = f"""
You are an AI office assistant.

Your job is to detect one or more user intents and return ONLY valid JSON.
Do not include markdown, code fences, or extra text.

Available actions:
- create
- update
- delete
- clarify

Current task list:
{json.dumps(task_list)}

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
                    user_id=user.id
                )
                db.add(db_task)
                db.commit()
                db.refresh(db_task)

                results.append(
                    {
                        "action": "create",
                        "message": "Task created successfully",
                        "task": serialize_task(db_task)
                    }
                )

            elif action == "update":
                task_id = item.get("task_id")
                task = db.query(Task).filter(Task.id == task_id, Task.user_id == user.id).first()

                if not task:
                    raise HTTPException(status_code=404, detail="Task not found")

                task.title = item.get("title", task.title)
                task.deadline = item.get("deadline", task.deadline)
                task.priority = item.get("priority", task.priority)

                db.commit()
                db.refresh(task)

                results.append(
                    {
                        "action": "update",
                        "message": "Task updated successfully",
                        "task": serialize_task(task)
                    }
                )

            elif action == "delete":
                fallback_task = find_best_task_match(tasks, request.text)

                if not fallback_task:
                    raise HTTPException(status_code=404, detail="Task not found")

                task = db.query(Task).filter(Task.id == fallback_task.id, Task.user_id == user.id).first()

                if not task:
                    raise HTTPException(status_code=404, detail="Task not found")

                deleted_task = serialize_task(task)

                db.delete(task)
                db.commit()

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
                deadline=task.get("deadline", ""),
                priority=task.get("priority", ""),
                user_id=user.id
            )
            db.add(db_task)
            db.commit()
            db.refresh(db_task)

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