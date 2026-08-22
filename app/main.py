from fastapi import FastAPI
from pydantic import BaseModel
from app.generator import generate_answer

app = FastAPI(
    title="AuditMind AI",
    description="Financial 10-K Audit & Compliance Assistant"
)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "AuditMind Engine is live and ready!"}

@app.post("/audit")
def audit_document(request: QueryRequest):
    response = generate_answer(request.question)
    return {
        "question": request.question,
        "audit_response": response
    }