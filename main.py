from fastapi import FastAPI, HTTPException
from openai import OpenAIError
from pydantic import BaseModel, Field

from graph import answer_question, get_api_key


app = FastAPI(
    title="Self-Correcting RAG & Web Fallback Assistant",
    description="Answers questions from a local policy knowledge base or web search.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be blank.")
    api_key = get_api_key()
    if not api_key or api_key == "your_api_key_here":
        raise HTTPException(
            status_code=503,
            detail="Set LLM_API_KEY in your .env file before asking a question.",
        )
    try:
        return answer_question(request.question.strip())
    except OpenAIError as error:
        raise HTTPException(
            status_code=502,
            detail=f"LLM request failed: {error}",
        ) from error
