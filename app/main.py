from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


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
    text = request.text

    return {
        "original_text": text,
        "summary": "This is a demo summary",
        "tasks": [
            {
                "title": "Follow up email",
                "deadline": "Tomorrow",
                "priority": "high"
            }
        ]
    }