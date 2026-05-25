"""Step 3 tests — chunk usage telemetry.

Covers:
- Chunk dataclass round-trip of the three new fields through both stores
- SQLite legacy-DB migration (no columns → columns present after init)
- `bump_chunk_access` primitive on both backends (idempotent batch +
  dedupe + unknown-id no-op + empty no-op)
- The five read-path MCP tools fire the hook end-to-end
- Telemetry never breaks the read path (store missing the method)
"""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

from vertical_brain.core.models import Chunk
from vertical_brain.mcp.server import VerticalBrainMCP
from vertical_brain.storage.json_store import JsonStore
from vertical_brain.storage.sqlite_store import SQLiteStore


# ── Helpers (mirror tests/test_mcp_server.py conventions) ────────────────────

def _mcp(tmp_path) -> tuple[VerticalBrainMCP, JsonStore]:
    store = JsonStore(tmp_path)
    return VerticalBrainMCP(store), store


def _call(mcp: VerticalBrainMCP, name: str, args: Optional[dict] = None) -> dict:
    return mcp.handle({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args or {}},
    })


# ── Round-trip ───────────────────────────────────────────────────────────────

def test_chunk_round_trips_new_fields_via_sqlite(tmp_path):
    store = SQLiteStore(tmp_path)
    saved = store.save_chunk(
        Chunk(
            node_path="WORK/Step3",
            content="usage telemetry chunk",
            access_count=7,
            last_accessed="2026-05-25T12:00:00+00:00",
            last_positive_use="2026-05-25T12:30:00+00:00",
        )
    )
    loaded = store.get_chunk(saved.id)
    assert loaded is not None
    assert loaded.access_count == 7
    assert loaded.last_accessed == "2026-05-25T12:00:00+00:00"
    assert loaded.last_positive_use == "2026-05-25T12:30:00+00:00"


def test_chunk_round_trips_new_fields_via_json_store(tmp_path):
    store = JsonStore(tmp_path)
    saved = store.save_chunk(
        Chunk(
            node_path="WORK/Step3",
            content="usage telemetry chunk",
            access_count=3,
            last_accessed="2026-05-25T12:00:00+00:00",
        )
    )
    loaded = store.get_chunk(saved.id)
    assert loaded is not None
    assert loaded.access_count == 3
    assert loaded.last_accessed == "2026-05-25T12:00:00+00:00"
    assert loaded.last_positive_use is None


def test_chunk_defaults_are_zero_and_none(tmp_path):
    store = SQLiteStore(tmp_path)
    saved = store.save_chunk(Chunk(node_path="WORK/Step3", content="plain"))
    loaded = store.get_chunk(saved.id)
    assert loaded is not None
    assert loaded.access_count == 0
    assert loaded.last_accessed is None
    assert loaded.last_positive_use is None


# ── Legacy SQLite migration ──────────────────────────────────────────────────

def test_sqlite_migration_adds_missing_columns(tmp_path):
    """Build a chunks table without the three new columns, then open it via
    SQLiteStore and confirm `_migrate_chunk_columns` filled them in.
    """
    db_path = tmp_path / "vertical_brain.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY, path TEXT NOT NULL, name TEXT NOT NULL,
            parent_path TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY, node_path TEXT NOT NULL, content TEXT NOT NULL,
            layer TEXT NOT NULL, content_type TEXT NOT NULL, status TEXT NOT NULL,
            source TEXT NOT NULL, confidence REAL NOT NULL, lineage_json TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE links (
            id TEXT PRIMARY KEY, source_path TEXT NOT NULL, target_path TEXT NOT NULL,
            link_type TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL,
            UNIQUE(source_path, target_path, link_type)
        );
        """
    )
    conn.commit()
    conn.close()

    SQLiteStore(tmp_path)  # opens db_path, runs migrations

    conn = sqlite3.connect(db_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    conn.close()
    assert "access_count" in cols
    assert "last_accessed" in cols
    assert "last_positive_use" in cols


# ── bump_chunk_access primitive ──────────────────────────────────────────────

def test_bump_chunk_access_sqlite_increments_and_dedupes(tmp_path):
    store = SQLiteStore(tmp_path)
    c1 = store.save_chunk(Chunk(node_path="WORK/A", content="one"))
    c2 = store.save_chunk(Chunk(node_path="WORK/B", content="two"))

    # Duplicate c1 in the input → still counts as one bump (set semantics).
    store.bump_chunk_access([c1.id, c1.id, c2.id], "2026-05-25T10:00:00+00:00")
    assert store.get_chunk(c1.id).access_count == 1  # type: ignore[union-attr]
    assert store.get_chunk(c2.id).access_count == 1  # type: ignore[union-attr]

    # Second batch with c1 alone → c1 to 2, c2 unchanged.
    store.bump_chunk_access([c1.id], "2026-05-25T10:05:00+00:00")
    c1_after = store.get_chunk(c1.id)
    c2_after = store.get_chunk(c2.id)
    assert c1_after is not None and c1_after.access_count == 2
    assert c1_after.last_accessed == "2026-05-25T10:05:00+00:00"
    assert c2_after is not None and c2_after.access_count == 1
    assert c2_after.last_accessed == "2026-05-25T10:00:00+00:00"


def test_bump_chunk_access_json_increments_and_dedupes(tmp_path):
    store = JsonStore(tmp_path)
    c1 = store.save_chunk(Chunk(node_path="WORK/A", content="one"))
    c2 = store.save_chunk(Chunk(node_path="WORK/B", content="two"))

    store.bump_chunk_access([c1.id, c1.id, c2.id], "2026-05-25T10:00:00+00:00")
    assert store.get_chunk(c1.id).access_count == 1  # type: ignore[union-attr]
    assert store.get_chunk(c2.id).access_count == 1  # type: ignore[union-attr]

    store.bump_chunk_access([c1.id], "2026-05-25T10:05:00+00:00")
    c1_after = store.get_chunk(c1.id)
    assert c1_after is not None and c1_after.access_count == 2
    assert c1_after.last_accessed == "2026-05-25T10:05:00+00:00"


def test_bump_chunk_access_ignores_unknown_and_empty(tmp_path):
    store = SQLiteStore(tmp_path)
    c1 = store.save_chunk(Chunk(node_path="WORK/A", content="one"))

    # Unknown id mixed with a real one — only the real one bumps.
    store.bump_chunk_access(
        [c1.id, "00000000-0000-0000-0000-000000000000"],
        "2026-05-25T10:00:00+00:00",
    )
    assert store.get_chunk(c1.id).access_count == 1  # type: ignore[union-attr]

    # Empty list and empty strings — no-op, no exception.
    store.bump_chunk_access([], "2026-05-25T10:01:00+00:00")
    store.bump_chunk_access(["", None], "2026-05-25T10:02:00+00:00")  # type: ignore[list-item]
    assert store.get_chunk(c1.id).access_count == 1  # type: ignore[union-attr]


# ── Read-path hook fires from each tool ──────────────────────────────────────

def test_list_chunks_bumps_each_returned_chunk(tmp_path):
    mcp, store = _mcp(tmp_path)
    a = store.save_chunk(Chunk(node_path="WORK/Step3", content="A"))
    b = store.save_chunk(Chunk(node_path="WORK/Step3", content="B"))

    _call(mcp, "list_chunks", {"path": "WORK/Step3"})

    assert store.get_chunk(a.id).access_count == 1  # type: ignore[union-attr]
    assert store.get_chunk(b.id).access_count == 1  # type: ignore[union-attr]


def test_list_chunks_bumps_only_filtered_active_chunks(tmp_path):
    mcp, store = _mcp(tmp_path)
    active = store.save_chunk(Chunk(node_path="WORK/Step3", content="A"))
    stale = store.save_chunk(
        Chunk(node_path="WORK/Step3", content="B", status="stale")
    )

    _call(mcp, "list_chunks", {"path": "WORK/Step3"})

    assert store.get_chunk(active.id).access_count == 1  # type: ignore[union-attr]
    assert store.get_chunk(stale.id).access_count == 0  # type: ignore[union-attr]


def test_get_chunk_bumps_returned_chunk(tmp_path):
    mcp, store = _mcp(tmp_path)
    c = store.save_chunk(Chunk(node_path="WORK/Step3", content="A"))

    _call(mcp, "get_chunk", {"chunk_id": c.id})

    assert store.get_chunk(c.id).access_count == 1  # type: ignore[union-attr]


def test_get_chunk_does_not_bump_when_returning_null(tmp_path):
    mcp, store = _mcp(tmp_path)
    stale = store.save_chunk(
        Chunk(node_path="WORK/Step3", content="A", status="stale")
    )

    _call(mcp, "get_chunk", {"chunk_id": stale.id})  # returns null
    assert store.get_chunk(stale.id).access_count == 0  # type: ignore[union-attr]

    _call(mcp, "get_chunk", {"chunk_id": stale.id, "include_stale": True})
    assert store.get_chunk(stale.id).access_count == 1  # type: ignore[union-attr]


def test_read_context_bumps_locked_items(tmp_path):
    mcp, store = _mcp(tmp_path)
    c = store.save_chunk(Chunk(node_path="WORK/Step3", content="A"))

    _call(mcp, "read_context", {"path": "WORK/Step3"})

    assert store.get_chunk(c.id).access_count == 1  # type: ignore[union-attr]


def test_context_search_bumps_returned_chunk_ids(tmp_path):
    mcp, store = _mcp(tmp_path)
    c = store.save_chunk(
        Chunk(node_path="WORK/Step3", content="usage tracking hit", layer="bronze")
    )

    resp = _call(mcp, "context_search", {"query": "usage tracking"})
    # Sanity: candidate_handles surfaced the chunk so a real hit happened.
    payload = json.loads(resp["result"]["content"][0]["text"])
    handle_ids = {h["chunk_id"] for h in payload.get("candidate_handles", [])}
    assert c.id in handle_ids

    assert store.get_chunk(c.id).access_count == 1  # type: ignore[union-attr]


# ── Telemetry never breaks the read path ─────────────────────────────────────

class _NoBumpJsonStore(JsonStore):
    """Storage backend without `bump_chunk_access` — simulates an older
    extension store. The read tools must still work.
    """

    def __init__(self, tmp_path):
        super().__init__(tmp_path)

    bump_chunk_access = None  # type: ignore[assignment]


def test_read_tools_survive_storage_missing_bump_method(tmp_path):
    store = _NoBumpJsonStore(tmp_path)
    mcp = VerticalBrainMCP(store)
    c = store.save_chunk(Chunk(node_path="WORK/Step3", content="A"))

    resp = _call(mcp, "list_chunks", {"path": "WORK/Step3"})
    chunks = json.loads(resp["result"]["content"][0]["text"])
    assert len(chunks) == 1
    assert chunks[0]["id"] == c.id

    resp = _call(mcp, "get_chunk", {"chunk_id": c.id})
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert payload is not None and payload["chunk_id"] == c.id
