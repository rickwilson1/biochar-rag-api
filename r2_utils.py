# r2_utils.py
import os
import boto3
from pathlib import Path

R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_KEY = os.environ["R2_KEY"]
R2_SECRET = os.environ["R2_SECRET"]
R2_BUCKET = os.environ.get("R2_BUCKET", "biochar-rag")

LOCAL_DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_KEY,
        aws_secret_access_key=R2_SECRET,
    )


def ensure_file_from_r2(key: str, local_name: str | None = None) -> Path:
    """
    Ensure a given object key from R2 exists locally.
    Returns the local Path.
    """
    if local_name is None:
        local_name = key.split("/")[-1]

    local_path = LOCAL_DATA_DIR / local_name
    if local_path.exists():
        return local_path

    s3 = get_s3_client()
    s3.download_file(R2_BUCKET, key, str(local_path))
    return local_path