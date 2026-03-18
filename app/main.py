import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class EmailRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "Buro Assistant API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}


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

    return {
        "original_text": request.text,
        "summary": parsed.get("summary", ""),
        "tasks": parsed.get("tasks", [])
    }