# r2_utils.py
import os
import boto3
from pathlib import Path

# Environment variables from Render
R2_ENDPOINT = os.environ.get("R2_ENDPOINT", "")
R2_KEY = os.environ.get("R2_KEY", "")
R2_SECRET = os.environ.get("R2_SECRET", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "biochar-rag")

# Use /var/data on Render, fallback to ./data locally
DATA_DIR = os.getenv("DATA_DIR", "/var/data")
LOCAL_DATA_DIR = Path(DATA_DIR)
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_s3_client():
    """
    Create a boto3 client for Cloudflare R2.
    Requires region_name="auto" to avoid SSL handshake errors.
    """
    if not R2_ENDPOINT or not R2_KEY or not R2_SECRET:
        raise ValueError("R2_ENDPOINT, R2_KEY, and R2_SECRET environment variables are required")
    
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_KEY,
        aws_secret_access_key=R2_SECRET,
        region_name="auto",
        config=boto3.session.Config(signature_version="s3v4"),
    )


def ensure_file_from_r2(key: str, local_name: str | None = None) -> Path:
    """
    Download an object from R2 to a local path (if not already present).
    """
    if local_name is None:
        local_name = key.split("/")[-1]

    local_path = LOCAL_DATA_DIR / local_name

    # If file is already downloaded, return it
    if local_path.exists():
        print(f"DEBUG: File already exists at {local_path}", flush=True)
        return local_path

    # Download from R2
    print(f"DEBUG: Downloading {key} from R2 to {local_path}", flush=True)
    s3 = get_s3_client()
    s3.download_file(R2_BUCKET, key, str(local_path))
    print(f"DEBUG: Download complete: {local_path}", flush=True)

    return local_path
