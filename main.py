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
        "chunks.parquet": (DATA_DIR / "chunks.parquet").exists(),
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
    allowed = {"embeddings.npy", "faiss.index", "emails_clean_normalized.csv", "chunks.parquet", "attachments.tar.gz"}
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
# Extract attachments archive
# ---------------------------------------------------------------------------

@app.post("/extract-attachments")
async def extract_attachments(authorization: str = Header(None)):
    """
    Extract the uploaded attachments.tar.gz archive.
    Requires Authorization header with your TOGETHER_API_KEY.
    """
    import tarfile
    import shutil
    
    # Verify auth
    if not UPLOAD_SECRET:
        raise HTTPException(status_code=500, detail="Server not configured")
    
    expected = f"Bearer {UPLOAD_SECRET}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid authorization")
    
    archive_path = DATA_DIR / "attachments.tar.gz"
    if not archive_path.exists():
        raise HTTPException(status_code=404, detail="attachments.tar.gz not found. Upload it first.")
    
    extract_dir = DATA_DIR / "attachments"
    
    try:
        # Remove existing attachments directory if present
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        
        # Extract the archive
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
        
        # Count extracted files
        file_count = len(list(extract_dir.glob("*"))) if extract_dir.exists() else 0
        
        # Optionally delete the archive to save space
        archive_path.unlink()
        
        return {
            "status": "extracted",
            "files": file_count,
            "path": str(extract_dir),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")


# ---------------------------------------------------------------------------
# Disk usage and cleanup
# ---------------------------------------------------------------------------

@app.get("/disk-usage")
def disk_usage():
    """Show disk usage for data directory."""
    import shutil
    
    total, used, free = shutil.disk_usage(DATA_DIR)
    
    # List all files and directories with sizes
    items = {}
    if DATA_DIR.exists():
        for item in DATA_DIR.iterdir():
            if item.is_file():
                items[item.name] = f"{item.stat().st_size / (1024*1024):.1f} MB"
            elif item.is_dir():
                # Sum up directory size
                dir_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                items[item.name + "/"] = f"{dir_size / (1024*1024):.1f} MB"
    
    return {
        "total_gb": f"{total / (1024**3):.2f} GB",
        "used_gb": f"{used / (1024**3):.2f} GB", 
        "free_gb": f"{free / (1024**3):.2f} GB",
        "data_dir_contents": items,
    }


@app.post("/cleanup")
async def cleanup_files(authorization: str = Header(None)):
    """
    Delete old/temporary files to free up disk space.
    Removes: attachments.tar.gz, old embeddings, etc.
    """
    import shutil
    
    if not UPLOAD_SECRET:
        raise HTTPException(status_code=500, detail="Server not configured")
    
    expected = f"Bearer {UPLOAD_SECRET}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid authorization")
    
    deleted = []
    
    # Delete tar.gz if exists
    tar_path = DATA_DIR / "attachments.tar.gz"
    if tar_path.exists():
        tar_path.unlink()
        deleted.append("attachments.tar.gz")
    
    # Delete old attachments directory if it exists
    att_dir = DATA_DIR / "attachments"
    if att_dir.exists():
        shutil.rmtree(att_dir)
        deleted.append("attachments/")
    
    return {"deleted": deleted, "message": "Cleanup complete"}


# ---------------------------------------------------------------------------
# Copyright protection - block high-risk PDFs
# ---------------------------------------------------------------------------

# Filename patterns indicating academic publisher PDFs
BLOCKED_FILENAME_PATTERNS = [
    r"^1-s2\.0-S",           # Elsevier/ScienceDirect
    r"^s4\d+-0",             # Springer
    r"^pii[_-]",             # Publisher Item Identifier
    r"^10\.\d+[_-]",         # DOI-based filenames
    r"-main[_-][a-f0-9]+\.pdf$",  # Common journal download pattern
    r"^nature\d+",           # Nature journals
    r"^srep\d+",             # Scientific Reports
    r"gcb[_-]?bioenergy",    # Global Change Biology Bioenergy
    r"bioresource[_-]?tech",  # Bioresource Technology
    r"^agronomy-\d+",        # MDPI Agronomy
    r"^sustainability-\d+",  # MDPI Sustainability
    r"^energies-\d+",        # MDPI Energies
    r"soil[_-]?biol",        # Soil Biology journals
    r"geoderma",             # Geoderma journal
    r"chemosphere",          # Chemosphere journal
    r"j\.envman",            # Journal of Environmental Management
    r"j\.soilbio",           # Journal of Soil Biology
]

# Publishers and copyright phrases to check in content
COPYRIGHT_PUBLISHERS = [
    "elsevier", "springer", "wiley", "nature publishing",
    "taylor & francis", "taylor and francis", "ieee",
    "american chemical society", "acs publications",
    "mdpi", "frontiers", "sage publications", "oxford university press",
    "cambridge university press", "royal society of chemistry",
    "cell press", "science magazine", "aaas",
]

COPYRIGHT_PHRASES = [
    "all rights reserved",
    "no part of this publication may be reproduced",
    "unauthorized reproduction",
    "licensed under cc by-nc",  # Non-commercial Creative Commons
    "for personal use only",
]


def is_copyright_blocked(filename: str, chunk_text: str = "") -> tuple[bool, str]:
    """
    Check if a PDF should be blocked due to copyright concerns.
    Returns (is_blocked, reason).
    """
    import re
    
    filename_lower = filename.lower()
    
    # Only check PDFs
    if not filename_lower.endswith('.pdf'):
        return False, ""
    
    # Check filename patterns
    for pattern in BLOCKED_FILENAME_PATTERNS:
        if re.search(pattern, filename_lower):
            return True, "This appears to be a copyrighted academic publication based on its filename pattern."
    
    # Check content for copyright notices
    if chunk_text:
        text_lower = chunk_text.lower()
        
        # Check for publisher names with copyright context
        for publisher in COPYRIGHT_PUBLISHERS:
            if publisher in text_lower:
                # Look for copyright indicators near publisher name
                if any(phrase in text_lower for phrase in ["©", "copyright", "published by", "journal of"]):
                    return True, f"This document appears to be published by a commercial publisher and may be copyrighted."
        
        # Check for explicit copyright phrases
        for phrase in COPYRIGHT_PHRASES:
            if phrase in text_lower:
                return True, "This document contains copyright restrictions."
    
    return False, ""


def get_attachment_text(filename: str) -> str:
    """
    Get the indexed text for an attachment from chunks.parquet.
    DISABLED: Loading 76MB parquet on each request causes memory issues.
    Using filename-based blocking only.
    """
    return ""  # Skip content checking to avoid memory issues


# ---------------------------------------------------------------------------
# Serve attachment files
# ---------------------------------------------------------------------------

@app.get("/attachment/{filename:path}")
def get_attachment(filename: str):
    """
    Serve an attachment file (PDF, JPG, PNG, etc.)
    Blocks copyrighted academic PDFs.
    """
    from fastapi.responses import FileResponse, JSONResponse
    import mimetypes
    
    # Sanitize filename to prevent directory traversal
    safe_filename = Path(filename).name
    file_path = DATA_DIR / "attachments" / safe_filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Attachment not found")
    
    # Check for copyright restrictions on PDFs
    if safe_filename.lower().endswith('.pdf'):
        chunk_text = get_attachment_text(safe_filename)
        is_blocked, reason = is_copyright_blocked(safe_filename, chunk_text)
        
        if is_blocked:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Download blocked",
                    "reason": reason,
                    "message": "This PDF cannot be downloaded due to potential copyright restrictions. "
                               "The content is available for viewing in the search results above.",
                    "filename": safe_filename,
                }
            )
    
    # Determine content type
    content_type, _ = mimetypes.guess_type(str(file_path))
    if content_type is None:
        content_type = "application/octet-stream"
    
    return FileResponse(
        path=file_path,
        media_type=content_type,
        filename=safe_filename,
    )


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
