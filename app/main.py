from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Buro Assistant API is running 🚀"}


@app.get("/health")
def health():
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