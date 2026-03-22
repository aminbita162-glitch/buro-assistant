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


@app.get("/")
def root():
    return {"message": "Buro Assistant API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def get_tasks():
    db = SessionLocal()
    tasks = db.query(Task).all()
    result = []

    for task in tasks:
        result.append(
            {
                "id": task.id,
                "title": task.title,
                "deadline": task.deadline,
                "priority": task.priority,
            }
        )

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

    result = {
        "id": task.id,
        "title": task.title,
        "deadline": task.deadline,
        "priority": task.priority,
    }

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

        saved_tasks.append(
            {
                "id": db_task.id,
                "title": db_task.title,
                "deadline": db_task.deadline,
                "priority": db_task.priority
            }
        )

    db.close()

    return {
        "original_text": request.text,
        "summary": parsed.get("summary", ""),
        "created_tasks": saved_tasks
    }