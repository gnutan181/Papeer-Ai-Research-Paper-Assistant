"""Object storage adapters; local storage remains the no-credential default."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol


class AssetStore(Protocol):
    def put(
        self, data: bytes, key: str | None = None, content_type: str = "image/png"
    ) -> str: ...
    def get(self, key: str) -> bytes: ...
    def delete(self, key: str) -> None: ...
    def get_access_url(self, key: str) -> str: ...


class LocalAssetStore:
    def __init__(self, root: str = "./data/assets") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents and path != self.root:
            raise ValueError("Invalid asset key")
        return path

    def put(
        self, data: bytes, key: str | None = None, content_type: str = "image/png"
    ) -> str:
        del content_type
        key = key or f"{hashlib.sha256(data).hexdigest()}.bin"
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def get_access_url(self, key: str) -> str:
        return self._path(key).as_uri()


class R2AssetStore:
    def __init__(
        self,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        bucket: str,
        public_base_url: str = "",
    ) -> None:
        import boto3

        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )
        self.bucket, self.public_base_url = bucket, public_base_url.rstrip("/")

    def put(
        self, data: bytes, key: str | None = None, content_type: str = "image/png"
    ) -> str:
        key = key or f"{hashlib.sha256(data).hexdigest()}.bin"
        self.client.put_object(
            Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
        )
        return key

    def get(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def get_access_url(self, key: str) -> str:
        if self.public_base_url:
            return f"{self.public_base_url}/{key}"
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=300
        )
