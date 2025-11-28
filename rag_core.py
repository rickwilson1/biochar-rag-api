# rag_core.py
from __future__ import annotations

import os
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
import requests

from r2_utils import ensure_file_from_r2
from email_store import get_full_email

TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]
TOGETHER_MODEL = os.environ.get("TOGETHER_MODEL", "deepseek-r1")  # adjust if needed

# --- Load FAISS index + embeddings at startup ---

_embeddings: np.ndarray | None = None
_index: faiss.Index | None = None
_metadata_df = None  # you can optionally load a DataFrame for chunk metadata


def load_vector_store():
    global _embeddings, _index

    if _index is not None and _embeddings is not None:
        return

    emb_path = ensure_file_from_r2("embeddings.npy")  # key name in R2
    idx_path = ensure_file_from_r2("faiss.index")

    _embeddings = np.load(str(emb_path))
    _index = faiss.read_index(str(idx_path))

def embed_query_locally(text: str) -> np.ndarray:
    """
    TEMPORARY STUB: Create a random embedding with correct dimension.
    Allows the RAG service to function until a real embedding model is added.
    """
    load_vector_store()

    dim = _index.d  # embedding dimension used in your FAISS index
    # return a 1 x dim vector
    return np.random.rand(1, dim).astype(np.float32)

def search(
    query: str,
    min_score: float = 0.65,
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    load_vector_store()

    # TODO: use your real embedding model here
    q_vec = embed_query_locally(query)  # shape (1, dim)
    scores, idxs = _index.search(q_vec, max_results)

    # scores are distances; you likely converted them to cosine sims earlier.
    # For now we'll just create placeholder scores.
    results: List[Dict[str, Any]] = []
    for rank, (dist, idx) in enumerate(zip(scores[0], idxs[0])):
        if idx < 0:
            continue

        # Convert distance to pseudo-score (you can replace with your real formula)
        score = float(1.0 / (1.0 + dist))

        if score < min_score:
            continue

        # At this point you’d pull metadata for the chunk: thread_id, subject, etc.
        # I’ll return minimal fields and assume you’ll hook in your real metadata.
        results.append(
            {
                "chunk_id": f"chunk-{idx}",
                "thread_id": f"thread-{idx}",  # replace with real thread id
                "score": score,
                "content_type": "email",
                "subject": f"Subject placeholder {idx}",
                "from": "",
                "to": "",
                "date": "",
                "text": f"Placeholder text for chunk {idx}",
            }
        )

    return results


def build_answer_from_together(
    query: str,
    sources: List[Dict[str, Any]],
) -> str:
    """
    Call Together chat/completions with your context. Adjust prompt & API path
    to match your current usage.
    """
    context_blocks = []
    for i, s in enumerate(sources, start=1):
        context_blocks.append(f"[Source {i}] {s.get('text', '')[:1200]}")

    context = "\n\n".join(context_blocks)

    prompt = f"""You are an assistant answering questions about biochar using the Biochar Groups.io archive.

Question:
{query}

Context (email excerpts with numbered sources):
{context}

When you cite, use [Source N] notation in the answer.
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
    # Adjust depending on Together’s response format
    return out["output"]["choices"][0]["text"]


def build_answer_payload(
    query: str,
    min_score: float,
    max_results: int,
) -> Dict[str, Any]:
    sources = search(query, min_score=min_score, max_results=max_results)
    answer = build_answer_from_together(query, sources)
    payload: Dict[str, Any] = {
        "answer": answer,
        "count": len(sources),
        "sources": sources,
        "stats": {
            "total_chunks": len(sources),
            "total_threads": len({s.get("thread_id") for s in sources}),
            "returned_threads": len({s.get("thread_id") for s in sources}),
        },
    }
    return payload
