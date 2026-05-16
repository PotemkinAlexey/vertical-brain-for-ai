"""Thread-local SQLite connection proxy.

Each thread that accesses the proxy gets its own SQLiteStore instance backed
by the same database file.  This avoids the sqlite3 same-thread restriction
while giving every thread a clean connection object with WAL pragmas applied.

Usage::

    store = ThreadLocalSQLiteStoreProxy("data/")
    # Use exactly like SQLiteStore from any thread.
    store.ensure_node("WORK/Project")
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any


class ThreadLocalSQLiteStoreProxy:
    """Proxy that exposes a per-thread SQLiteStore instance.

    All attribute accesses and method calls are forwarded to the thread-local
    store, which is lazily created on the first access from each thread.
    """

    def __init__(self, root: str | Path = "data", db_name: str = "vertical_brain.sqlite") -> None:
        self._root = Path(root)
        self._db_name = db_name
        self._local = threading.local()

    def _store(self) -> Any:
        if not hasattr(self._local, "store"):
            from vertical_brain.storage.sqlite_store import SQLiteStore
            self._local.store = SQLiteStore(self._root, self._db_name)
        return self._local.store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store(), name)

    def __repr__(self) -> str:
        return (
            f"ThreadLocalSQLiteStoreProxy(root={self._root!r}, "
            f"db_name={self._db_name!r})"
        )
