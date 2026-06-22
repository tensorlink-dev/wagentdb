"""SQLite metadata database that lives in the object store.

The SQLite file is the source of truth for *structured* data (projects, runs,
metrics, the graph, ...). It is small relative to the artifacts, so we keep the
whole file in the object store under ``settings.db_key``:

* on open, if a local working copy doesn't exist but the object does, we
  download it;
* after each write transaction (when ``auto_sync`` is on) we upload the file
  back to the object store.

This is a deliberately simple, single-writer model. It is robust for one agent
(or one server process) writing at a time, which matches how training runs log.
For concurrent writers, run the FastAPI server and have agents talk to it over
HTTP so writes are serialized in one process.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import Settings
from .objectstore import ObjectStore

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = "1"


class Database:
    def __init__(
        self,
        settings: Settings,
        object_store: Optional[ObjectStore] = None,
    ):
        self.settings = settings
        self.object_store = object_store
        self._lock = threading.RLock()

        cache_dir = settings.resolved_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.local_path = cache_dir / Path(settings.db_key).name

        self._maybe_download()

        self.conn = sqlite3.connect(str(self.local_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()

    # ------------------------------------------------------------------ sync
    def _maybe_download(self) -> None:
        """Pull the sqlite file from the object store if we don't have it yet."""
        if self.object_store is None:
            return
        if self.local_path.exists():
            return
        if self.object_store.exists(self.settings.db_key):
            self.object_store.get_file(self.settings.db_key, str(self.local_path))

    def sync_up(self) -> None:
        """Upload the current sqlite file to the object store."""
        if self.object_store is None:
            return
        with self._lock:
            self.object_store.put_file(
                self.settings.db_key,
                str(self.local_path),
                content_type="application/x-sqlite3",
            )

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(SCHEMA_PATH.read_text())
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
            self.conn.commit()
        # Persist the freshly created file so the object store has it.
        if self.object_store is not None and not self.object_store.exists(self.settings.db_key):
            self.sync_up()

    # --------------------------------------------------------------- queries
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, tuple(params))

    def write(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        """Execute a mutating statement, commit, and sync to the object store."""
        with self._lock:
            cur = self.conn.execute(sql, tuple(params))
            self.conn.commit()
        if self.settings.auto_sync:
            self.sync_up()
        return cur

    def write_many(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        with self._lock:
            self.conn.executemany(sql, [tuple(r) for r in rows])
            self.conn.commit()
        if self.settings.auto_sync:
            self.sync_up()

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[Dict[str, Any]]:
        row = self.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def query_all(self, sql: str, params: Iterable[Any] = ()) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.execute(sql, params).fetchall()]

    def close(self) -> None:
        with self._lock:
            self.conn.commit()
            self.conn.close()
        if self.settings.auto_sync:
            self.sync_up()
