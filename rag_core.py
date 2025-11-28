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

TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]
TOGETHER_MODEL = os.environ.get("TOGETHER_MODEL", "deepseek-r1")


# ---------------------------------------------------------------------------
# Global vector store handles
# ---------------------------------------------------------------------------

_embeddings: np.ndarray | None = None
_index: faiss.Index | None = None


# ---------------------------------------------------------------------------
# Load FAISS + embeddings from Render mounted disk
# ---------------------------------------------------------------------------

def load_vector_store():
    global _embeddings, _index

    if _index is not None and _embeddings is not None:
        return

    from pathlib import Path

    emb_path = Path("/opt/render/project/src/embeddings.npy")
    idx_path = Path("/opt/render/project/src/faiss.index")

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings at {emb_path}")

    if not idx_path.exists():
        raise FileNotFoundError(f"Missing FAISS index at {idx_path}")

    _embeddings = np.load(str(emb_path))
    _index = faiss.read_index(str(idx_path))


# ---------------------------------------------------------------------------
# Temporary stub embedding for queries
# ---------------------------------------------------------------------------

def embed_query_locally(text: str) -> np.ndarray:
    """
    Temporary deterministic pseudo-embedding.
    Replace with the real embedding model you used to generate embeddings.npy.
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
# FAISS search
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
# Call Together API
# ---------------------------------------------------------------------------

def build_answer_from_together(
    query: str,
    sources: List[Dict[str, Any]],
) -> str:

    context_blocks: List[str] = []
    for i, s in enumerate(sources, start=1):
        snippet = s.get("text", "")
        context_blocks.append(f"[Source {i}] {snippet[:1200]}")

    context = "\n\n".join(context_blocks)

    prompt = f"""You are an assistant answering questions about biochar using the Biochar Groups.io archive.

Question:
{query}

Context (email excerpts):
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
       
