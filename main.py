# main.py
from __future__ import annotations

import asyncio
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_core import build_answer_payload, search
from email_store import get_full_email

app = FastAPI(title="Biochar RAG API")


# ---------------------------------------------------------------------------
# Request models for proper validation
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    min_score: float = 0.65
    max_results: int = 20


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Search endpoint - semantic search only
# ---------------------------------------------------------------------------

@app.post("/search")
def search_endpoint(payload: QueryRequest) -> Dict[str, Any]:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    results = search(payload.query, min_score=payload.min_score, max_results=payload.max_results)
    return {"results": results}


# ---------------------------------------------------------------------------
# Query endpoint - search + LLM answer
# ---------------------------------------------------------------------------

@app.post("/query")
def query_endpoint(payload: QueryRequest) -> Dict[str, Any]:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    return build_answer_payload(payload.query, min_score=payload.min_score, max_results=payload.max_results)


# ---------------------------------------------------------------------------
# Query stream endpoint - streaming response
# ---------------------------------------------------------------------------

@app.post("/query_stream")
async def query_stream_endpoint(payload: QueryRequest):
    """
    Streaming endpoint that yields the answer in chunks.
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    data = build_answer_payload(payload.query, min_score=payload.min_score, max_results=payload.max_results)
    answer = data["answer"]

    async def stream_response():
        # Yield in chunks for streaming effect
        for chunk in answer.split(" "):
            yield chunk + " "
            await asyncio.sleep(0.01)

    return StreamingResponse(stream_response(), media_type="text/plain")


# ---------------------------------------------------------------------------
# Email endpoint - get full email by thread_id
# ---------------------------------------------------------------------------

@app.get("/email/{thread_id}")
def get_email(thread_id: str) -> Dict[str, Any]:
    email = get_full_email(thread_id)
    if email is None:
        raise HTTPException(status_code=404, detail="Email not found")
    return email
