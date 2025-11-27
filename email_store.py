# email_store.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd

EMAIL_DATA_PATH = Path(os.environ.get("EMAIL_DATA_PATH", "data/emails_clean_normalized.csv"))

_email_df: pd.DataFrame | None = None


def load_email_dataframe() -> pd.DataFrame:
    global _email_df
    if _email_df is None:
        if not EMAIL_DATA_PATH.exists():
            raise FileNotFoundError(f"Email data file not found at {EMAIL_DATA_PATH}")
        _email_df = pd.read_csv(EMAIL_DATA_PATH)
    return _email_df


def get_full_email(thread_id: str) -> Dict[str, Any] | None:
    """
    Return a dict with fields used by the UI:
      full_text, subject, from, to, date, thread_id, chunk_count
    Adjust column names to match your CSV.
    """
    df = load_email_dataframe()
    row = df.loc[df["thread_id"] == thread_id]
    if row.empty:
        return None

    r = row.iloc[0]
    return {
        "thread_id": thread_id,
        "subject": r.get("subject", ""),
        "from": r.get("from", ""),
        "to": r.get("to", ""),
        "date": r.get("date", ""),
        "full_text": r.get("full_text", r.get("text", "")),
        "chunk_count": int(r.get("chunk_count", 1)),
    }