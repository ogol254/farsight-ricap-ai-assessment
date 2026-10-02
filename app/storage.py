"""Private photo persistence; no user-controlled paths or public bucket."""
import os
from pathlib import Path
from uuid import uuid4
import httpx


class PhotoStore:
    def __init__(self, root="artifacts/photos"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.key = os.getenv("SUPABASE_SERVICE_KEY", "")
        self.bucket = "ricap-photos"
        if bool(self.url) != bool(self.key):
            raise RuntimeError("Both Supabase URL and server-only service key are required")

    @property
    def remote(self):
        return bool(self.url and self.key)

    def headers(self):
        headers = {"apikey": self.key}
        if not self.key.startswith("sb_secret_"):
            headers["Authorization"] = "Bearer " + self.key
        return headers

    def put(self, image):
        key = str(uuid4()) + ".jpg"
        if self.remote:
            response = httpx.post(f"{self.url}/storage/v1/object/{self.bucket}/{key}", headers={**self.headers(), "Content-Type": "image/jpeg"}, content=image, timeout=25)
            response.raise_for_status()
        else:
            (self.root / key).write_bytes(image)
        return key

    def get(self, key):
        if Path(key).name != key or not key.endswith(".jpg"):
            raise ValueError("Invalid stored photo key")
        if self.remote:
            response = httpx.get(f"{self.url}/storage/v1/object/authenticated/{self.bucket}/{key}", headers=self.headers(), timeout=25)
            response.raise_for_status()
            return response.content
        return (self.root / key).read_bytes()
