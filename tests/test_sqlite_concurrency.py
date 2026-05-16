"""Smoke tests for SQLite WAL-mode concurrency."""
from __future__ import annotations

import threading

from vertical_brain.core.models import Chunk
from vertical_brain.storage.sqlite_store import SQLiteStore


def test_wal_mode_is_enabled(tmp_path):
    store = SQLiteStore(tmp_path)
    row = store.conn.execute("PRAGMA journal_mode").fetchone()
    assert row[0] == "wal"


def test_two_stores_same_db_reader_sees_writer(tmp_path):
    """One connection writes; a second connection opened after the write reads it."""
    writer = SQLiteStore(tmp_path)
    writer.save_chunk(Chunk(node_path="WORK/A", content="written by store 1"))

    reader = SQLiteStore(tmp_path)
    chunks = reader.get_chunks_by_path("WORK/A")
    assert len(chunks) == 1
    assert chunks[0].content == "written by store 1"


def test_concurrent_writes_from_two_threads(tmp_path):
    """Two threads each writing through their own SQLiteStore instance complete without error."""
    errors: list[Exception] = []

    def write(i: int) -> None:
        try:
            store = SQLiteStore(tmp_path)
            for j in range(5):
                store.save_chunk(Chunk(node_path=f"WORK/T{i}", content=f"fact {j}"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent writes raised: {errors}"

    # Verify all 10 chunks landed.
    reader = SQLiteStore(tmp_path)
    total = len(reader.list_chunks())
    assert total == 10
