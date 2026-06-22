"""Configuration for wagentdb.

Settings can be provided explicitly or read from the environment. The two
backends are:

* ``local`` - object store is a directory on disk; the SQLite file lives there
  too. Great for development, tests, and single-machine use.
* ``r2``    - object store is a Cloudflare R2 bucket (S3-compatible). The SQLite
  file is kept in the bucket and cached locally; large blobs (artifacts) live
  as objects in the same bucket.

Relevant environment variables::

    WAGENTDB_BACKEND        local | r2            (default: local)
    WAGENTDB_LOCAL_DIR      path for local backend (default: ./.wagentdb)
    WAGENTDB_DB_KEY         object key for the sqlite file (default: wagentdb.sqlite)
    WAGENTDB_CACHE_DIR      local cache dir for the r2-backed sqlite file

    # R2 / S3-compatible credentials (R2_* preferred, AWS_* honored as fallback)
    WAGENTDB_R2_BUCKET / R2_BUCKET
    WAGENTDB_R2_ENDPOINT / R2_ENDPOINT_URL
    WAGENTDB_R2_ACCOUNT_ID / R2_ACCOUNT_ID
    WAGENTDB_R2_ACCESS_KEY_ID / R2_ACCESS_KEY_ID / AWS_ACCESS_KEY_ID
    WAGENTDB_R2_SECRET_ACCESS_KEY / R2_SECRET_ACCESS_KEY / AWS_SECRET_ACCESS_KEY
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    return default


@dataclass
class Settings:
    backend: str = "local"
    # Local backend / cache locations
    local_dir: Path = Path(".wagentdb")
    cache_dir: Optional[Path] = None
    # Key of the sqlite file inside the object store
    db_key: str = "wagentdb.sqlite"
    # R2 / S3
    r2_bucket: Optional[str] = None
    r2_endpoint_url: Optional[str] = None
    r2_account_id: Optional[str] = None
    r2_access_key_id: Optional[str] = None
    r2_secret_access_key: Optional[str] = None
    r2_region: str = "auto"
    # If True the sqlite file is uploaded to the object store after every write.
    auto_sync: bool = True

    def __post_init__(self) -> None:
        self.local_dir = Path(self.local_dir)
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
        # Derive the R2 endpoint from the account id if not given explicitly.
        if self.backend == "r2" and not self.r2_endpoint_url and self.r2_account_id:
            self.r2_endpoint_url = f"https://{self.r2_account_id}.r2.cloudflarestorage.com"

    @classmethod
    def from_env(cls) -> "Settings":
        backend = _env("WAGENTDB_BACKEND", default="local") or "local"
        cache_dir = _env("WAGENTDB_CACHE_DIR")
        return cls(
            backend=backend,
            local_dir=Path(_env("WAGENTDB_LOCAL_DIR", default=".wagentdb")),
            cache_dir=Path(cache_dir) if cache_dir else None,
            db_key=_env("WAGENTDB_DB_KEY", default="wagentdb.sqlite"),
            r2_bucket=_env("WAGENTDB_R2_BUCKET", "R2_BUCKET"),
            r2_endpoint_url=_env("WAGENTDB_R2_ENDPOINT", "R2_ENDPOINT_URL"),
            r2_account_id=_env("WAGENTDB_R2_ACCOUNT_ID", "R2_ACCOUNT_ID"),
            r2_access_key_id=_env(
                "WAGENTDB_R2_ACCESS_KEY_ID", "R2_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"
            ),
            r2_secret_access_key=_env(
                "WAGENTDB_R2_SECRET_ACCESS_KEY",
                "R2_SECRET_ACCESS_KEY",
                "AWS_SECRET_ACCESS_KEY",
            ),
            r2_region=_env("WAGENTDB_R2_REGION", default="auto") or "auto",
        )

    def resolved_cache_dir(self) -> Path:
        """Where the working copy of the sqlite file lives on disk."""
        if self.cache_dir:
            return self.cache_dir
        return self.local_dir / "cache"
