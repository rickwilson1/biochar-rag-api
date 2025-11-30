# email_store.py
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Dict, Any

# Use /var/data on Render, fallback to ./data locally
DATA_DIR = os.getenv("DATA_DIR", "/opt/render/project/src/data")
EMAIL_DB_PATH = Path(f"{DATA_DIR}/emails.db")
EMAIL_CSV_PATH = Path(f"{DATA_DIR}/emails_clean_normalized.csv")

_db_initialized = False


def _get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory for dict-like access."""
    conn = sqlite3.connect(str(EMAIL_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db_from_csv():
    """Convert CSV to SQLite database if needed (one-time operation)."""
    global _db_initialized
    
    if _db_initialized:
        return
    
    # If DB exists and is recent, skip conversion
    if EMAIL_DB_PATH.exists():
        db_mtime = EMAIL_DB_PATH.stat().st_mtime
        if EMAIL_CSV_PATH.exists():
            csv_mtime = EMAIL_CSV_PATH.stat().st_mtime
            if db_mtime >= csv_mtime:
                print(f"DEBUG: Using existing SQLite database at {EMAIL_DB_PATH}", flush=True)
                _db_initialized = True
                return
    
    # Convert CSV to SQLite
    if not EMAIL_CSV_PATH.exists():
        print(f"WARNING: No CSV file at {EMAIL_CSV_PATH}, database will be empty", flush=True)
        _db_initialized = True
        return
    
    print(f"DEBUG: Converting CSV to SQLite database...", flush=True)
    
    import pandas as pd
    
    # Read CSV in chunks to avoid memory spike
    chunk_size = 10000
    first_chunk = True
    total_rows = 0
    
    conn = sqlite3.connect(str(EMAIL_DB_PATH))
    
    for chunk in pd.read_csv(EMAIL_CSV_PATH, chunksize=chunk_size):
        chunk.to_sql('emails', conn, if_exists='replace' if first_chunk else 'append', index=False)
        total_rows += len(chunk)
        first_chunk = False
        print(f"DEBUG: Processed {total_rows} rows...", flush=True)
    
    # Create index on thread_id for fast lookups
    conn.execute("CREATE INDEX IF NOT EXISTS idx_thread_id ON emails(thread_id)")
    conn.commit()
    conn.close()
    
    print(f"DEBUG: Created SQLite database with {total_rows} emails", flush=True)
    _db_initialized = True


def get_full_email(thread_id: str) -> Dict[str, Any] | None:
    """
    Return a dict with fields used by the UI:
      full_text, subject, from, to, date, thread_id, chunk_count
    Uses SQLite for memory-efficient lookups.
    """
    _init_db_from_csv()
    
    if not EMAIL_DB_PATH.exists():
        return None
    
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM emails WHERE thread_id = ? LIMIT 1",
            (thread_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        # Convert Row to dict
        row_dict = dict(row)
        
        return {
            "thread_id": thread_id,
            "subject": row_dict.get("subject", ""),
            "from": row_dict.get("from", ""),
            "to": row_dict.get("to", ""),
            "cc": row_dict.get("cc", ""),
            "date": row_dict.get("date", ""),
            "full_text": row_dict.get("clean_text", row_dict.get("normalized_text", "")),
            "chunk_count": 1,
        }
    finally:
        conn.close()
