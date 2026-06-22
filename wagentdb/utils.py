"""Small shared helpers: id generation, time, json."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import uuid
from typing import Any, Dict, Iterable, Optional, Tuple


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


def sha256_file(path: str, chunk: int = 1 << 20) -> Tuple[str, int]:
    """Stream a file to compute (sha256, size) without loading it into memory."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def flatten_numeric(
    data: Dict[str, Any], parent: str = "", sep: str = "/"
) -> Iterable[Tuple[str, float]]:
    """Flatten a (possibly nested) dict into ``key -> float`` pairs.

    Nested dicts become ``parent/child`` keys. Booleans count as numbers.
    Non-numeric values (strings, lists, None) and non-finite floats (NaN/inf)
    are skipped, so any framework's log dict can be passed through safely.
    """
    for key, value in data.items():
        name = f"{parent}{sep}{key}" if parent else str(key)
        if isinstance(value, dict):
            yield from flatten_numeric(value, name, sep)
        elif isinstance(value, bool):
            yield name, float(value)
        elif isinstance(value, (int, float)):
            f = float(value)
            if math.isfinite(f):
                yield name, f
