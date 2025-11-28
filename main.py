# main.py
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_core import build_answer_payload, search
from email_store import get_full_email

app = FastAPI(title="Biochar RAG API")

DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data"))
UPLOAD_SECRET = os.getenv("TOGETHER_API_KEY", "")  # Reuse this as upload auth


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
# Data status - check which files are present
# ---------------------------------------------------------------------------

@app.get("/data-status")
def data_status() -> Dict[str, Any]:
    """Check which data files exist on the persistent disk."""
    files = {
        "embeddings.npy": (DATA_DIR / "embeddings.npy").exists(),
        "faiss.index": (DATA_DIR / "faiss.index").exists(),
        "emails_clean_normalized.csv": (DATA_DIR / "emails_clean_normalized.csv").exists(),
    }
    
    sizes = {}
    for name, exists in files.items():
        if exists:
            size_mb = (DATA_DIR / name).stat().st_size / (1024 * 1024)
            sizes[name] = f"{size_mb:.1f} MB"
        else:
            sizes[name] = "missing"
    
    return {
        "data_dir": str(DATA_DIR),
        "files": files,
        "sizes": sizes,
        "ready": all(files.values()),
    }


# ---------------------------------------------------------------------------
# Upload endpoint - for uploading data files to persistent disk
# ---------------------------------------------------------------------------

@app.post("/upload/{filename}")
async def upload_file(
    filename: str,
    file: UploadFile = File(...),
    authorization: str = Header(None),
):
    """
    Upload a data file to the persistent disk.
    Requires Authorization header with your TOGETHER_API_KEY.
    
    Usage:
        curl -X POST "https://your-app.onrender.com/upload/embeddings.npy" \
             -H "Authorization: Bearer YOUR_TOGETHER_API_KEY" \
             -F "file=@embeddings.npy"
    """
    # Verify auth
    if not UPLOAD_SECRET:
        raise HTTPException(status_code=500, detail="Server not configured for uploads")
    
    expected = f"Bearer {UPLOAD_SECRET}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid authorization")
    
    # Only allow specific filenames
    allowed = {"embeddings.npy", "faiss.index", "emails_clean_normalized.csv"}
    if filename not in allowed:
        raise HTTPException(status_code=400, detail=f"Filename must be one of: {allowed}")
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save file in chunks to handle large files
    file_path = DATA_DIR / filename
    total_bytes = 0
    
    with open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            f.write(chunk)
            total_bytes += len(chunk)
    
    size_mb = total_bytes / (1024 * 1024)
    return {
        "status": "uploaded",
        "filename": filename,
        "size": f"{size_mb:.1f} MB",
        "path": str(file_path),
    }


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
