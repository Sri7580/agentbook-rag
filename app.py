"""FastAPI backend. Run: uvicorn app:app --reload"""
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.graph import build_rag_graph

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["graph"] = build_rag_graph(config.PINECONE_INDEX_NAME)
    yield
    _state.clear()


app = FastAPI(title="Agentic AI RAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[str]
    confidence_score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")
    result = _state["graph"].invoke(
        {"question": query, "context": [], "answer": "", "score": 0.0}
    )
    return QueryResponse(
        answer=result["answer"],
        retrieved_chunks=result["context"],
        confidence_score=result["score"],
    )
