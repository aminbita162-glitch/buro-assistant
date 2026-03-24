import json
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

from sqlalchemy import create_engine, Column, Integer, String, or_
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
    return FileResponse("app/static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def get_stats():
    db = SessionLocal()
    tasks = db.query(Task).order_by(Task.id.desc()).all()

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

    db.close()

    return {
        "total_tasks": total,
        "high_priority_tasks": high_count,
        "medium_priority_tasks": medium_count,
        "low_priority_tasks": low_count,
        "recent_tasks": recent_tasks
    }


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


@app.get("/tasks/search")
def search_tasks(
    q: str = Query(default=""),
    priority: str = Query(default=""),
    deadline: str = Query(default="")
):
    db = SessionLocal()
    query = db.query(Task)

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
    db.refresh(task)
    result = serialize_task(task)
    db.close()

    return {
        "message": "Task updated successfully",
        "task": result
    }


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

Rules:
- Choose exactly one best matching task_id.
- Return only one task_id.

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
        db.close()
        raise HTTPException(
            status_code=500,
            detail="AI response was not valid JSON"
        )

    if isinstance(parsed, list):
        actions = parsed
    else:
        actions = parsed.get("actions", [])

    if not actions:
        db.close()
        raise HTTPException(status_code=400, detail="No actions returned by AI")

    for item in actions:
        if item.get("action") == "clarify":
            db.close()
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
                priority=item.get("priority", "medium")
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
            task = db.query(Task).filter(Task.id == task_id).first()

            if not task:
                db.close()
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
            task_id = item.get("task_id")
            task = db.query(Task).filter(Task.id == task_id).first()

            if not task:
                db.close()
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
            db.close()
            raise HTTPException(status_code=400, detail="Invalid action returned by AI")

    db.close()

    if len(results) == 1:
        return results[0]

    return {
        "message": "Multiple actions completed successfully",
        "results": results
    }


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