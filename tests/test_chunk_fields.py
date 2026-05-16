import json
import sqlite3

from vertical_brain.core.models import Chunk
from vertical_brain.storage.json_store import JsonStore
from vertical_brain.storage.sqlite_store import SQLiteStore


def test_content_hash_is_deterministic():
    a = Chunk(node_path="WORK/A", content="hello world")
    b = Chunk(node_path="WORK/B", content="hello world")
    assert a.content_hash == b.content_hash
    assert len(a.content_hash) == 64


def test_content_hash_normalizes_whitespace():
    a = Chunk(node_path="WORK/A", content="hello  world")
    b = Chunk(node_path="WORK/A", content="hello world")
    c = Chunk(node_path="WORK/A", content="  hello\tworld\n")
    assert a.content_hash == b.content_hash == c.content_hash


def test_chunk_key_is_optional():
    chunk = Chunk(node_path="WORK/A", content="x")
    assert chunk.chunk_key is None


def test_valid_from_set_on_creation():
    chunk = Chunk(node_path="WORK/A", content="x")
    assert chunk.valid_from
    assert chunk.valid_to is None
    assert chunk.supersedes == []


def test_old_json_chunk_without_new_fields_loads(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/A")
    legacy_row = {
        "node_path": "WORK/A",
        "content": "legacy content",
        "layer": "bronze",
        "content_type": "note",
        "status": "active",
        "source": "manual",
        "confidence": 1.0,
        "lineage": [],
        "id": "legacy-id",
        "created_at": "2020-01-01T00:00:00+00:00",
        "updated_at": "2020-01-01T00:00:00+00:00",
    }
    store.chunks_file.write_text(json.dumps([legacy_row]), encoding="utf-8")

    chunks = store.list_chunks()
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_key is None
    assert chunk.supersedes == []
    assert chunk.valid_from == "2020-01-01T00:00:00+00:00"
    assert chunk.valid_to is None
    assert chunk.content_hash  # computed in __post_init__


def test_old_sqlite_chunk_migrates_automatically(tmp_path):
    db_file = tmp_path / "vertical_brain.sqlite"
    conn = sqlite3.connect(db_file)
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            node_path TEXT NOT NULL,
            content TEXT NOT NULL,
            layer TEXT NOT NULL,
            content_type TEXT NOT NULL,
            status TEXT NOT NULL,
            source TEXT NOT NULL,
            confidence REAL NOT NULL,
            lineage_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "old-id", "WORK/A", "old content", "bronze", "note", "active",
            "manual", 1.0, "[]", "2020-01-01T00:00:00+00:00",
            "2020-01-01T00:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()

    store = SQLiteStore(tmp_path)
    chunks = store.list_chunks()
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_key is None
    assert chunk.supersedes == []
    assert chunk.valid_from == "2020-01-01T00:00:00+00:00"
    assert chunk.valid_to is None


def test_json_store_persists_new_chunk_fields(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/A")
    chunk = Chunk(
        node_path="WORK/A",
        content="versioned",
        chunk_key="my-key",
        supersedes=["older-id"],
        valid_to="2030-01-01T00:00:00+00:00",
    )
    store.save_chunk(chunk)

    reloaded = JsonStore(tmp_path).list_chunks()[0]
    assert reloaded.chunk_key == "my-key"
    assert reloaded.supersedes == ["older-id"]
    assert reloaded.valid_to == "2030-01-01T00:00:00+00:00"
    assert reloaded.content_hash == chunk.content_hash


def test_sqlite_store_persists_new_chunk_fields(tmp_path):
    store = SQLiteStore(tmp_path)
    store.ensure_node("WORK/A")
    chunk = Chunk(
        node_path="WORK/A",
        content="versioned",
        chunk_key="my-key",
        supersedes=["older-id"],
        valid_to="2030-01-01T00:00:00+00:00",
    )
    store.save_chunk(chunk)
    store.conn.close()

    reloaded = SQLiteStore(tmp_path).list_chunks()[0]
    assert reloaded.chunk_key == "my-key"
    assert reloaded.supersedes == ["older-id"]
    assert reloaded.valid_to == "2030-01-01T00:00:00+00:00"
    assert reloaded.content_hash == chunk.content_hash
