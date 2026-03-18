from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze")
def analyze():
    return {
        "summary": "This is a sample summary",
        "tasks": [
            {
                "title": "Sample task",
                "deadline": "Not specified",
                "priority": "medium"
            }
        ]
    }