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
TOGETHER_MODEL = os.environ.get("TOGETHER_MODEL", "deepseek-r1")  # adjust if needed

# ---------------------------------------------------------------------------
# Global vector store handles
# ---------------------------------------------------------------------------

_embeddings: np.ndarray | None = None
_index: faiss.Index | None = None
_metadata_df = None  # placeholder if you later attach real metadata


# ---------------------------------------------------------------------------
# Load FAISS index + embeddings
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
# Query embedding (TEMPORARY STUB)
# ---------------------------------------------------------------------------

def embed_query_locally(text: str) -> np.ndarray:
    """
    Temporary fake embedding so the API works end-to-end.

    It produces a deterministic pseudo-random vector of the same
    dimension as your stored embeddings. This lets FAISS search run
    without having a real embedding model wired up yet.

    TODO: replace with a real embedding model (e.g. BGE-large) that
    matches how you built `embeddings.npy`.
    """
    # Make sure vectors are loaded so we know the dimensionality.
    if _embeddings is None:
        load_vector_store()

    assert _embeddings is not None
    dim = _embeddings.shape[1]

    # Deterministic RNG keyed on the query text
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)

    vec = rng.random(dim)  # random vector
    # Normalize so magnitude is 1 (like a unit embedding)
    vec = vec / np.linalg.norm(vec)

    # FAISS expects shape (1, dim) for a single query
    return vec.reshape(1, -1)


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def search(
    query: str,
    min_score: float = 0.65,
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    """
    Run a similarity search in the FAISS index and return a list of source dicts.

    Each result currently contains placeholder metadata except for the
    score and a synthetic chunk/thread id. You can later wire this into
    your real email metadata (subjects, senders, etc.).
    """
    load_vector_store()
    assert _index is not None

    q_vec = embed_query_locally(query)  # shape (1, dim)
    distances, idxs = _index.search(q_vec, max_results)

    results: List[Dict[str, Any]] = []

    # Convert FAISS distances into a pseudo-similarity score.
    for rank, (dist, idx) in enumerate(zip(distances[0], idxs[0])):
        if idx < 0:
            continue

        # Simple monotonic transform: smaller distance ⇒ higher score
        score = float(1.0 / (1.0 + dist))

        if score < min_score:
            continue

        # TODO: replace with real metadata lookup from your email store
        results.append(
            {
                "rank": rank,
                "chunk_id": f"chunk-{idx}",
                "thread_id": f"thread-{idx}",
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


# ---------------------------------------------------------------------------
# Call Together for final answer
# ---------------------------------------------------------------------------

def build_answer_from_together(
    query: str,
    sources: List[Dict[str, Any]],
) -> str:
    """
    Call Together's completion endpoint with the retrieved context.
    Adjust `model` and response parsing to match your actual usage.
    """
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
    
# Adjust this if Together changes their schema
# Try the common Together schema first
if "output" in out and "choices" in out["output"]:
    return out["output"]["choices"][0]["text"]

# Fallback schema
if "choices" in out:
    return out["choices"][0]["text"]

# If neither works, fail loudly with a helpful error
raise KeyError(f"Unexpected Together response format: {out}")


# ---------------------------------------------------------------------------
# Main API payload builder
# ---------------------------------------------------------------------------

def build_answer_payload(
    query: str,
    min_score: float,
    max_results: int,
) -> Dict[str, Any]:
    """
    Orchestrate search + LLM answer into a JSON-serializable payload
    consumed by your Streamlit frontend.
    """
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
