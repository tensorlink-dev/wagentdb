"""Pluggable object storage.

The rest of wagentdb only ever talks to the :class:`ObjectStore` interface, so
the same code runs against a local directory (dev/tests) or a Cloudflare R2
bucket (production). R2 is S3-compatible, so the R2 backend is a thin wrapper
around boto3.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from .config import Settings


class ObjectStore(ABC):
    """Minimal blob store: put/get bytes & files, list, delete, sign."""

    @abstractmethod
    def put_bytes(self, key: str, data: bytes, content_type: Optional[str] = None) -> None: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def list(self, prefix: str = "") -> List[str]: ...

    def put_file(self, key: str, path: str, content_type: Optional[str] = None) -> None:
        with open(path, "rb") as fh:
            self.put_bytes(key, fh.read(), content_type=content_type)

    def get_file(self, key: str, path: str) -> None:
        data = self.get_bytes(key)
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(data)

    def presigned_url(self, key: str, expires: int = 3600) -> Optional[str]:
        """Return a temporary download URL, or ``None`` if unsupported."""
        return None


class LocalObjectStore(ObjectStore):
    """Stores objects as files under a root directory."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Keys may contain '/'; treat them as nested paths but keep them inside root.
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError(f"key escapes object store root: {key!r}")
        return p

    def put_bytes(self, key: str, data: bytes, content_type: Optional[str] = None) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish write.
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(data)
        shutil.move(str(tmp), str(p))

    def get_bytes(self, key: str) -> bytes:
        with open(self._path(key), "rb") as fh:
            return fh.read()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()

    def list(self, prefix: str = "") -> List[str]:
        out: List[str] = []
        for path in self.root.rglob("*"):
            if path.is_file() and not path.name.endswith(".tmp"):
                rel = str(path.relative_to(self.root))
                if rel.startswith(prefix):
                    out.append(rel)
        return sorted(out)

    def put_file(self, key: str, path: str, content_type: Optional[str] = None) -> None:
        dst = self._path(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)

    def get_file(self, key: str, path: str) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(key), dest)

    def presigned_url(self, key: str, expires: int = 3600) -> Optional[str]:
        p = self._path(key)
        return p.as_uri() if p.exists() else None


class R2ObjectStore(ObjectStore):
    """Cloudflare R2 (S3-compatible) backend, via boto3."""

    def __init__(self, settings: Settings):
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError as exc:  # pragma: no cover - exercised only without boto3
            raise ImportError(
                "The R2 backend requires boto3. Install with `pip install wagentdb[r2]`."
            ) from exc

        if not settings.r2_bucket:
            raise ValueError("R2 backend requires a bucket (WAGENTDB_R2_BUCKET).")
        if not settings.r2_endpoint_url:
            raise ValueError(
                "R2 backend requires an endpoint or account id "
                "(WAGENTDB_R2_ENDPOINT or WAGENTDB_R2_ACCOUNT_ID)."
            )

        self.bucket = settings.r2_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint_url,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name=settings.r2_region,
            config=BotoConfig(signature_version="s3v4"),
        )

    def put_bytes(self, key: str, data: bytes, content_type: Optional[str] = None) -> None:
        extra = {"ContentType": content_type} if content_type else {}
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, **extra)

    def get_bytes(self, key: str) -> bytes:
        resp = self.client.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def put_file(self, key: str, path: str, content_type: Optional[str] = None) -> None:
        # boto3 upload_file streams and multiparts large files automatically.
        extra = {"ContentType": content_type} if content_type else None
        self.client.upload_file(path, self.bucket, key, ExtraArgs=extra)

    def get_file(self, key: str, path: str) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(dest))

    def list(self, prefix: str = "") -> List[str]:
        keys: List[str] = []
        token = None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self.client.list_objects_v2(**kwargs)
            keys.extend(obj["Key"] for obj in resp.get("Contents", []))
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return keys

    def presigned_url(self, key: str, expires: int = 3600) -> Optional[str]:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )


def build_object_store(settings: Settings) -> ObjectStore:
    if settings.backend == "r2":
        return R2ObjectStore(settings)
    if settings.backend == "local":
        return LocalObjectStore(settings.local_dir / "objects")
    raise ValueError(f"unknown backend: {settings.backend!r}")
