# main.py
from __future__ import annotations

import asyncio
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from rag_core import build_answer_payload, search
from email_store import get_full_email

app = FastAPI(title="Biochar RAG API")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/search")
def search_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    query: str = payload.get("query", "")
    min_score: float = float(payload.get("min_score", 0.65))
    max_results: int = int(payload.get("max_results", 20))

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    results = search(query, min_score=min_score, max_results=max_results)
    return {"results": results}


@app.post("/query")
def query_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    query: str = payload.get("query", "")
    min_score: float = float(payload.get("min_score", 0.65))
    max_results: int = int(payload.get("max_results", 20))

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    return build_answer_payload(query, min_score=min_score, max_results=max_results)


@app.post("/query_stream")
async def query_stream_endpoint(payload: Dict[str, Any]):
    """
    For now, we implement a simple non-streaming wrapper that yields the full
    answer once. You can later switch to Together's streaming API.
    """
    query: str = payload.get("query", "")
    min_score: float = float(payload.get("min_score", 0.65))
    max_results: int = int(payload.get("max_results", 20))

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    data = build_answer_payload(query, min_score=min_score, max_results=max_results)
    answer = data["answer"]

    async def fake_stream():
        # Yield in chunks just so the UI sees multiple events
        for chunk in answer.split(" "):
            yield chunk + " "
            await asyncio.sleep(0.01)

    return StreamingResponse(fake_stream(), media_type="text/plain")


@app.get("/email/{thread_id}")
def get_email(thread_id: str) -> Dict[str, Any]:
    email = get_full_email(thread_id)
    if email is None:
        raise HTTPException(status_code=404, detail="Email not found")
    return email