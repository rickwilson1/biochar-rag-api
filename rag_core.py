# rag_core.py
from __future__ import annotations

import os
from typing import List, Dict, Any

import numpy as np
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

TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "deepseek-r1")


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


    emb_path = Path("/opt/render/project/src/data/embeddings.npy")
    idx_path = Path("/opt/render/project/src/data/faiss.index")

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings at {emb_path}")

    if not idx_path.exists():
        raise FileNotFoundError(f"Missing FAISS index at {idx_path}")

    _embeddings = np.load(str(emb_path))
    _index = faiss.read_index(str(idx_path))


# ---------------------------------------------------------------------------
# TEMPORARY deterministic embedding stub
# ---------------------------------------------------------------------------

def embed_query_locally(text: str) -> np.ndarray:
    """
    Temporary deterministic pseudo-embedding.
    Replace with your real embedding model later.
    """
    if _embeddings is None:
        load_vector_store()

    dim = _embeddings.shape[1]

    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)

    vec = rng.random(dim)
    vec = vec / np.linalg.norm(vec)

    return vec.reshape(1, -1)


# ---------------------------------------------------------------------------
# FAISS similarity search
# ---------------------------------------------------------------------------

def search(
    query: str,
    min_score: float = 0.65,
    max_results: int = 20,
) -> List[Dict[str, Any]]:

    load_vector_store()
    q_vec = embed_query_locally(query)

    distances, idxs = _index.search(q_vec, max_results)

    results: List[Dict[str, Any]] = []

    for rank, (dist, idx) in enumerate(zip(distances[0], idxs[0])):
        if idx < 0:
            continue

        score = float(1.0 / (1.0 + dist))
        if score < min_score:
            continue

        results.append(
            {
                "rank": rank,
                "chunk_id": f"chunk-{idx}",
                "thread_id": f"thread-{idx}",
                "score": score,
                "content_type": "email",
                "subject": f"Subject {idx}",
                "from": "",
                "to": "",
                "date": "",
                "text": f"Placeholder text for chunk {idx}",
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
        "input": prompt,
        "max_tokens": 1024,
        "temperature": 0.3,
        "stream": False,
    }

    resp = requests.post(
        "https://api.together.xyz/v1/completions",
        json=data,
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()

    out = resp.json()

    # Handle both Together response formats
    if "output" in out and "choices" in out["output"]:
        return out["output"]["choices"][0]["text"]

    if "choices" in out:
        return out["choices"][0]["text"]

    raise KeyError(f"Unexpected Together response format: {out}")


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
