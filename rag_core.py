# rag_core.py
from __future__ import annotations

import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import faiss
import requests


# ---------------------------------------------------------------------------
# Together API configuration
# ---------------------------------------------------------------------------

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

# Debug: see whether Render actually passed it in
print(
    "DEBUG: TOGETHER_API_KEY is "
    + ("SET" if TOGETHER_API_KEY else "MISSING"),
    flush=True,
)

TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")


# ---------------------------------------------------------------------------
# Data paths - use /var/data on Render, fallback to ./data locally
# ---------------------------------------------------------------------------

DATA_DIR = os.getenv("DATA_DIR", "/opt/render/project/src/data")


# ---------------------------------------------------------------------------
# Global vector store handles
# ---------------------------------------------------------------------------

_embeddings: np.ndarray | None = None
_index: faiss.Index | None = None


# ---------------------------------------------------------------------------
# Load FAISS + embeddings from Render persistent disk
# ---------------------------------------------------------------------------

def load_vector_store():
    global _embeddings, _index

    if _index is not None and _embeddings is not None:
        return

    from pathlib import Path

    emb_path = Path(DATA_DIR) / "embeddings.npy"
    idx_path = Path(DATA_DIR) / "faiss.index"

    print(f"DEBUG: Loading embeddings from {emb_path}", flush=True)
    print(f"DEBUG: Loading FAISS index from {idx_path}", flush=True)

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings at {emb_path}")

    if not idx_path.exists():
        raise FileNotFoundError(f"Missing FAISS index at {idx_path}")

    # Use memory-mapped files to avoid loading everything into RAM
    # This is critical for Render Starter (512MB RAM) with large files
    _embeddings = np.load(str(emb_path), mmap_mode='r')
    _index = faiss.read_index(str(idx_path), faiss.IO_FLAG_MMAP)

    print(f"DEBUG: Loaded embeddings shape: {_embeddings.shape} (memory-mapped)", flush=True)
    print(f"DEBUG: FAISS index loaded successfully (memory-mapped)", flush=True)


# ---------------------------------------------------------------------------
# Embedding model configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


def embed_query(text: str) -> np.ndarray:
    """
    Get embedding for query text using Together AI's embedding API.
    Uses BAAI/bge-large-en-v1.5 model.
    """
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    data = {
        "model": EMBEDDING_MODEL,
        "input": text,
    }
    
    resp = requests.post(
        "https://api.together.xyz/v1/embeddings",
        json=data,
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    
    result = resp.json()
    embedding = result["data"][0]["embedding"]
    
    return np.array(embedding, dtype=np.float32).reshape(1, -1)


# ---------------------------------------------------------------------------
# FAISS similarity search
# ---------------------------------------------------------------------------

def search(
    query: str,
    min_score: float = 0.3,
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    from email_store import load_email_dataframe

    load_vector_store()
    q_vec = embed_query(query)

    distances, idxs = _index.search(q_vec, max_results)

    # Load email data for content retrieval
    df = load_email_dataframe()

    results: List[Dict[str, Any]] = []

    for rank, (dist, idx) in enumerate(zip(distances[0], idxs[0])):
        if idx < 0:
            continue

        score = float(1.0 / (1.0 + dist))
        if score < min_score:
            continue

        # Get email content from dataframe by index
        if idx < len(df):
            row = df.iloc[idx]
            subject = str(row.get("subject", "")) if "subject" in df.columns else ""
            from_addr = str(row.get("from", "")) if "from" in df.columns else ""
            to_addr = str(row.get("to", "")) if "to" in df.columns else ""
            date = str(row.get("date", "")) if "date" in df.columns else ""
            # Try multiple possible column names for email body
            text = ""
            for col in ["clean_text", "normalized_text", "full_text", "text", "body", "content"]:
                if col in df.columns and pd.notna(row.get(col)):
                    text = str(row.get(col))
                    break
            thread_id = str(row.get("thread_id", f"thread-{idx}")) if "thread_id" in df.columns else f"thread-{idx}"
        else:
            subject = f"Subject {idx}"
            from_addr = ""
            to_addr = ""
            date = ""
            text = f"Content for chunk {idx}"
            thread_id = f"thread-{idx}"

        results.append(
            {
                "rank": rank,
                "chunk_id": f"chunk-{idx}",
                "thread_id": thread_id,
                "score": score,
                "content_type": "email",
                "subject": subject,
                "from": from_addr,
                "to": to_addr,
                "date": date,
                "text": text[:1500] if text else "",  # Limit text length
            }
        )

    return results


# ---------------------------------------------------------------------------
# Call Together API to build final answer
# ---------------------------------------------------------------------------

def build_answer_from_together(
    query: str,
    sources: List[Dict[str, Any]],
) -> str:

    # Build context from source chunks
    context_blocks: List[str] = []
    for i, s in enumerate(sources, start=1):
        snippet = s.get("text", "") or ""
        context_blocks.append(f"[Source {i}] {snippet[:1200]}")

    context = "\n\n".join(context_blocks)

    prompt = f"""You are an assistant answering questions about biochar using the Biochar Groups.io archive.

Question:
{query}

Context (email excerpts with numbered sources):
{context}

When you cite, use [Source N].
"""

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": TOGETHER_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.3,
    }

    resp = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        json=data,
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()

    out = resp.json()
    return out["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Build final answer payload for API
# ---------------------------------------------------------------------------

def build_answer_payload(
    query: str,
    min_score: float,
    max_results: int,
) -> Dict[str, Any]:

    sources = search(query, min_score=min_score, max_results=max_results)
    answer = build_answer_from_together(query, sources)

    return {
        "answer": answer,
        "count": len(sources),
        "sources": sources,
        "stats": {
            "total_chunks": len(sources),
            "total_threads": len({s["thread_id"] for s in sources}),
            "returned_threads": len({s["thread_id"] for s in sources}),
        },
    }
