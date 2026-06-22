"""Small shared helpers: id generation, time, json."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import uuid
from typing import Any, Optional


def new_id(prefix: str) -> str:
    """Short, sortable-enough, human-readable id, e.g. ``run_3f9a1c2b8d44``."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def dumps(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    return json.dumps(obj, default=str, sort_keys=True)


def loads(text: Optional[str]) -> Any:
    if text is None or text == "":
        return None
    return json.loads(text)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
