"""Tests for the v1.12 `metadata` extension slot on Chunk and Node.

`metadata: dict[str, Any]` is an open-core extension point: the framework
neither reads nor interprets the dict; it just guarantees the bytes
round-trip through write/read across both storage backends and survive
SQLite schema migrations.

These tests pin the round-trip and migration contract so enterprise can
rely on `chunk.metadata` / `node.metadata` for classification,
tenant_id, retention policy, etc.
"""
from __future__ import annotations

import sqlite3

from vertical_brain.core.models import (
    Chunk,
    ChunkInput,
    Node,
    StorageOperation,
)
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.storage.json_store import JsonStore
from vertical_brain.storage.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


def test_chunk_metadata_defaults_to_empty_dict():
    assert Chunk(node_path="X", content="y").metadata == {}


def test_node_metadata_defaults_to_empty_dict():
    assert Node(path="X", name="X").metadata == {}


def test_chunk_input_metadata_defaults_to_empty_dict():
    assert ChunkInput(content="x").metadata == {}


# ---------------------------------------------------------------------------
# Round-trip through SQLiteStore
# ---------------------------------------------------------------------------


def test_sqlite_chunk_metadata_round_trip(tmp_path):
    store = SQLiteStore(tmp_path)
    meta = {"classification": "internal", "tenant_id": "acme-42", "tags": ["q4", "finance"]}
    chunk = store.save_chunk(Chunk(node_path="WORK/A", content="alpha", metadata=meta))

    fetched = store.get_chunk(chunk.id)
    assert fetched is not None
    assert fetched.metadata == meta


def test_sqlite_node_metadata_round_trip(tmp_path):
    store = SQLiteStore(tmp_path)
    store.ensure_node("WORK/A")
    node = store.get_node("WORK/A")
    assert node is not None
    node.metadata = {"geo_residency": "EU", "owner": "team-alpha"}
    store.update_node(node)

    refetched = store.get_node("WORK/A")
    assert refetched is not None
    assert refetched.metadata == {"geo_residency": "EU", "owner": "team-alpha"}


def test_sqlite_chunk_metadata_survives_update(tmp_path):
    store = SQLiteStore(tmp_path)
    chunk = store.save_chunk(Chunk(node_path="WORK/A", content="x", metadata={"k": "v1"}))
    chunk.metadata = {"k": "v2", "more": True}
    store.update_chunk(chunk)

    refetched = store.get_chunk(chunk.id)
    assert refetched is not None
    assert refetched.metadata == {"k": "v2", "more": True}


def test_sqlite_chunk_metadata_empty_by_default(tmp_path):
    store = SQLiteStore(tmp_path)
    chunk = store.save_chunk(Chunk(node_path="WORK/A", content="x"))
    refetched = store.get_chunk(chunk.id)
    assert refetched is not None
    assert refetched.metadata == {}


def test_sqlite_chunk_metadata_visible_via_list_and_path(tmp_path):
    """Metadata round-trips through both list_chunks and get_chunks_by_path —
    these are the high-fanout entry points used by every read-path API."""
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="a", metadata={"tag": "x"}))
    store.save_chunk(Chunk(node_path="WORK/A", content="b", metadata={"tag": "y"}))

    tags_list = sorted(c.metadata.get("tag") for c in store.list_chunks())
    tags_path = sorted(c.metadata.get("tag") for c in store.get_chunks_by_path("WORK/A"))
    assert tags_list == ["x", "y"]
    assert tags_path == ["x", "y"]


# ---------------------------------------------------------------------------
# Round-trip through JsonStore
# ---------------------------------------------------------------------------


def test_json_chunk_metadata_round_trip(tmp_path):
    store = JsonStore(tmp_path)
    meta = {"classification": "restricted", "owners": ["a@x.com"]}
    chunk = store.save_chunk(Chunk(node_path="WORK/A", content="alpha", metadata=meta))

    fetched = store.get_chunk(chunk.id)
    assert fetched is not None
    assert fetched.metadata == meta


def test_json_node_metadata_round_trip(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/A")
    node = store.get_node("WORK/A")
    assert node is not None
    node.metadata = {"region": "us-east-1"}
    store.update_node(node)

    refetched = store.get_node("WORK/A")
    assert refetched is not None
    assert refetched.metadata == {"region": "us-east-1"}


# ---------------------------------------------------------------------------
# ChunkInput → Chunk forwarding
# ---------------------------------------------------------------------------


def test_chunk_input_metadata_forwarded_through_append_chunk(tmp_path):
    store = SQLiteStore(tmp_path)
    ex = StorageOperationExecutor(store)
    op = StorageOperation(
        operation="append_chunk",
        target_path="WORK/A",
        chunk=ChunkInput(content="alpha", metadata={"classification": "confidential"}),
    )
    result = ex.apply(op)
    fetched = store.get_chunk(result.chunk_id)
    assert fetched is not None
    assert fetched.metadata == {"classification": "confidential"}


# ---------------------------------------------------------------------------
# Legacy DB migration: chunks/nodes table without metadata_json column
# ---------------------------------------------------------------------------


def _make_legacy_db(directory) -> None:
    """Hand-craft a pre-v1.12 SQLite database missing the metadata_json columns."""
    db_path = directory / "vertical_brain.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            parent_path TEXT,
            node_type TEXT NOT NULL DEFAULT 'default',
            gold_summary TEXT NOT NULL DEFAULT '',
            is_dirty INTEGER NOT NULL DEFAULT 0,
            version INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            node_path TEXT NOT NULL,
            content TEXT NOT NULL,
            layer TEXT NOT NULL DEFAULT 'bronze',
            content_type TEXT NOT NULL DEFAULT 'note',
            status TEXT NOT NULL DEFAULT 'active',
            source TEXT NOT NULL DEFAULT 'manual',
            confidence REAL NOT NULL DEFAULT 1.0,
            lineage_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT ''
        );
        INSERT INTO nodes (id, path, name, parent_path, created_at, updated_at)
            VALUES ('n1', 'WORK/A', 'A', 'WORK', '', '');
        INSERT INTO chunks (id, node_path, content, lineage_json, created_at, updated_at)
            VALUES ('c1', 'WORK/A', 'legacy content', '[]', '', '');
        """
    )
    conn.commit()
    conn.close()


def test_legacy_db_migrates_to_include_metadata_columns(tmp_path):
    _make_legacy_db(tmp_path)
    # Opening the store runs the migration.
    store = SQLiteStore(tmp_path)
    legacy_chunk = store.get_chunk("c1")
    legacy_node = store.get_node("WORK/A")

    assert legacy_chunk is not None
    assert legacy_chunk.metadata == {}
    assert legacy_node is not None
    assert legacy_node.metadata == {}

    # And new metadata can be written on the migrated DB.
    legacy_chunk.metadata = {"now": "filled"}
    store.update_chunk(legacy_chunk)
    refetched = store.get_chunk("c1")
    assert refetched.metadata == {"now": "filled"}
