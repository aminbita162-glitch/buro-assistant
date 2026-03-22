import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    deadline = Column(String)
    priority = Column(String)


Base.metadata.create_all(bind=engine)


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


def serialize_task(task: Task):
    return {
        "id": task.id,
        "title": task.title,
        "deadline": task.deadline,
        "priority": task.priority,
    }


@app.get("/")
def root():
    return {"message": "Buro Assistant API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def get_tasks():
    db = SessionLocal()
    tasks = db.query(Task).order_by(Task.id.desc()).all()
    result = [serialize_task(task) for task in tasks]
    db.close()

    return {
        "count": len(result),
        "tasks": result
    }


@app.get("/tasks/{task_id}")
def get_task(task_id: int):
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        db.close()
        raise HTTPException(status_code=404, detail="Task not found")

    result = serialize_task(task)
    db.close()
    return result


@app.delete("/tasks/{task_id}")
def delete_task(task_id: int):
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        db.close()
        raise HTTPException(status_code=404, detail="Task not found")

    db.delete(task)
    db.commit()
    db.close()

    return {"message": "Task deleted successfully"}


@app.put("/tasks/{task_id}")
def update_task(task_id: int, request: UpdateTaskRequest):
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        db.close()
        raise HTTPException(status_code=404, detail="Task not found")

    task.title = request.title
    task.deadline = request.deadline
    task.priority = request.priority

    db.commit()
    db.close()

    return {"message": "Task updated successfully"}


@app.post("/tasks/{task_id}/ai-update")
def ai_update_task(task_id: int, request: UpdateTaskAIRequest):
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        db.close()
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
        db.close()
        raise HTTPException(
            status_code=500,
            detail="AI response was not valid JSON"
        )

    task.title = parsed.get("title", task.title)
    task.deadline = parsed.get("deadline", task.deadline)
    task.priority = parsed.get("priority", task.priority)

    db.commit()
    db.refresh(task)

    result = {
        "message": "Task updated successfully",
        "task": serialize_task(task)
    }

    db.close()
    return result


@app.post("/tasks/ai-delete")
def ai_delete_task(request: DeleteTaskAIRequest):
    db = SessionLocal()
    tasks = db.query(Task).order_by(Task.id.desc()).all()

    if not tasks:
        db.close()
        raise HTTPException(status_code=404, detail="No tasks found")

    task_list = [serialize_task(task) for task in tasks]

    prompt = f"""
You are an AI office assistant.

The user wants to delete one task from the task list.
Return ONLY valid JSON.
Do not include markdown, code fences, or extra text.

Task list:
{json.dumps(task_list)}

User instruction:
{request.text}

Required JSON format:
{{
  "task_id": 123
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
        db.close()
        raise HTTPException(
            status_code=500,
            detail="AI response was not valid JSON"
        )

    task_id = parsed.get("task_id")

    if not task_id:
        db.close()
        raise HTTPException(status_code=400, detail="Task id was not returned by AI")

    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        db.close()
        raise HTTPException(status_code=404, detail="Task not found")

    deleted_task = serialize_task(task)

    db.delete(task)
    db.commit()
    db.close()

    return {
        "message": "Task deleted successfully",
        "deleted_task": deleted_task
    }


@app.post("/assistant")
def assistant(request: AssistantRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    db = SessionLocal()
    tasks = db.query(Task).order_by(Task.id.desc()).all()
    task_list = [serialize_task(task) for task in tasks]

    prompt = f"""
You are an AI office assistant.

Your job is to detect the user's intent and return ONLY valid JSON.
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
- Choose create only when the user wants a brand new task.
- Choose update only when the user clearly refers to an existing task and wants it changed.
- Choose delete only when the user clearly refers to an existing task and wants it removed.
- Choose clarify when the instruction is ambiguous, unclear, or there is not enough information.
- For create, set task_id to null and return title, deadline, and priority.
- For update, return the correct task_id and the updated title, deadline, and priority.
- For delete, return the correct task_id. Title, deadline, and priority can be empty strings.
- For clarify, set task_id to null and return a short question in clarify_message.
- Match against the current task list carefully. Prefer the best matching task title when the user refers to an existing task.
- priority must be one of: low, medium, high

Required JSON format:
{{
  "action": "create or update or delete or clarify",
  "task_id": 123,
  "title": "task title or empty string",
  "deadline": "deadline or Not specified or empty string",
  "priority": "low or medium or high or empty string",
  "clarify_message": "question or empty string"
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
        db.close()
        raise HTTPException(
            status_code=500,
            detail="AI response was not valid JSON"
        )

    action = parsed.get("action")

    if action == "clarify":
        db.close()
        return {
            "action": "clarify",
            "message": parsed.get("clarify_message", "Please clarify your request.")
        }

    if action == "create":
        db_task = Task(
            title=parsed.get("title", ""),
            deadline=parsed.get("deadline", "Not specified"),
            priority=parsed.get("priority", "medium")
        )
        db.add(db_task)
        db.commit()
        db.refresh(db_task)

        result = {
            "action": "create",
            "message": "Task created successfully",
            "task": serialize_task(db_task)
        }
        db.close()
        return result

    if action == "update":
        task_id = parsed.get("task_id")
        task = db.query(Task).filter(Task.id == task_id).first()

        if not task:
            db.close()
            raise HTTPException(status_code=404, detail="Task not found")

        task.title = parsed.get("title", task.title)
        task.deadline = parsed.get("deadline", task.deadline)
        task.priority = parsed.get("priority", task.priority)

        db.commit()
        db.refresh(task)

        result = {
            "action": "update",
            "message": "Task updated successfully",
            "task": serialize_task(task)
        }
        db.close()
        return result

    if action == "delete":
        task_id = parsed.get("task_id")
        task = db.query(Task).filter(Task.id == task_id).first()

        if not task:
            db.close()
            raise HTTPException(status_code=404, detail="Task not found")

        deleted_task = serialize_task(task)

        db.delete(task)
        db.commit()
        db.close()

        return {
            "action": "delete",
            "message": "Task deleted successfully",
            "task": deleted_task
        }

    db.close()
    raise HTTPException(status_code=400, detail="Invalid action returned by AI")


@app.post("/analyze")
def analyze(request: EmailRequest):
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
    saved_tasks = []

    for task in parsed.get("tasks", []):
        db_task = Task(
            title=task.get("title", ""),
            deadline=task.get("deadline", ""),
            priority=task.get("priority", "")
        )
        db.add(db_task)
        db.commit()
        db.refresh(db_task)

        saved_tasks.append(serialize_task(db_task))

    db.close()

    return {
        "original_text": request.text,
        "summary": parsed.get("summary", ""),
        "created_tasks": saved_tasks
    }