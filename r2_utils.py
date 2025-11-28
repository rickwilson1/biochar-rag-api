# r2_utils.py
import os
import boto3
from pathlib import Path

# Environment variables from Render
R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_KEY = os.environ["R2_KEY"]
R2_SECRET = os.environ["R2_SECRET"]
R2_BUCKET = os.environ.get("R2_BUCKET", "biochar-rag")

# Local data directory
LOCAL_DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_s3_client():
    """
    Create a boto3 client for Cloudflare R2.
    Requires region_name="auto" to avoid SSL handshake errors.
    """
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_KEY,
        aws_secret_access_key=R2_SECRET,
        region_name="auto",           # <-- REQUIRED FIX
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
        return local_path

    # Download from R2
    s3 = get_s3_client()
    s3.download_file(R2_BUCKET, key, str(local_path))

    return local_path
