# main.py
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_core import build_answer_payload, search
from email_store import get_full_email

app = FastAPI(title="Biochar RAG API")

DATA_DIR = Path(os.getenv("DATA_DIR", "/opt/render/project/src/data"))
UPLOAD_SECRET = os.getenv("TOGETHER_API_KEY", "")  # Reuse this as upload auth


# ---------------------------------------------------------------------------
# Request models for proper validation
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    min_score: float = 0.3
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
    
    # Check if data dir exists and is writable
    dir_exists = DATA_DIR.exists()
    dir_writable = False
    if dir_exists:
        try:
            test_file = DATA_DIR / ".write_test"
            test_file.touch()
            test_file.unlink()
            dir_writable = True
        except Exception:
            pass
    
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
        "dir_exists": dir_exists,
        "dir_writable": dir_writable,
        "files": files,
        "sizes": sizes,
        "ready": all(files.values()),
    }


# ---------------------------------------------------------------------------
# Upload endpoint - streams directly to disk (memory efficient)
# ---------------------------------------------------------------------------

@app.post("/upload/{filename}")
async def upload_file(
    filename: str,
    request: Request,
    authorization: str = Header(None),
):
    """
    Upload a data file to the persistent disk using streaming.
    Requires Authorization header with your TOGETHER_API_KEY.
    
    Usage:
        curl -X POST "https://your-app.onrender.com/upload/embeddings.npy" \
             -H "Authorization: Bearer YOUR_TOGETHER_API_KEY" \
             -H "Content-Type: application/octet-stream" \
             --data-binary @embeddings.npy
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
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot create data dir: {e}")
    
    # Stream directly to file - no memory buffering
    file_path = DATA_DIR / filename
    total_bytes = 0
    
    try:
        with open(file_path, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
                total_bytes += len(chunk)
    except Exception as e:
        # Clean up partial file
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
    
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
