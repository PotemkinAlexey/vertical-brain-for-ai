"""Namespace rename/move cascade tests."""
from __future__ import annotations

import pytest

from vertical_brain.core.models import LinkInput, StorageOperation
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.storage.json_store import JsonStore
from vertical_brain.storage.sqlite_store import SQLiteStore


def _populate(store, prefix: str) -> None:
    from vertical_brain.core.models import Chunk, ChunkInput
    store.ensure_node(f"{prefix}/Alpha")
    store.ensure_node(f"{prefix}/Beta")
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(
        operation="append_chunk",
        target_path=f"{prefix}/Alpha",
        chunk=ChunkInput(content="fact about alpha"),
    ))
    executor.apply(StorageOperation(
        operation="append_chunk",
        target_path=f"{prefix}/Beta",
        chunk=ChunkInput(content="fact about beta"),
    ))
    executor.apply(StorageOperation(
        operation="create_link",
        target_path=f"{prefix}/Alpha",
        links=[LinkInput(target_path=f"{prefix}/Beta", link_type="peer", reason="related")],
    ))


# ── JsonStore ─────────────────────────────────────────────────────────────────

def test_rename_cascade_moves_chunks_json(tmp_path):
    store = JsonStore(tmp_path)
    _populate(store, "OLD/Project")
    store.rename_namespace("OLD/Project", "NEW/Project")
    paths = {c.node_path for c in store.list_chunks()}
    assert all("OLD" not in p for p in paths)
    assert any("NEW/Project/Alpha" in p for p in paths)


def test_rename_cascade_moves_links_json(tmp_path):
    store = JsonStore(tmp_path)
    _populate(store, "OLD/Project")
    store.rename_namespace("OLD/Project", "NEW/Project")
    links = store.list_links()
    for link in links:
        assert "OLD" not in link.source_path
        assert "OLD" not in link.target_path


def test_rename_cascade_moves_nodes_json(tmp_path):
    store = JsonStore(tmp_path)
    _populate(store, "OLD/Project")
    store.rename_namespace("OLD/Project", "NEW/Project")
    node_paths = {n.path for n in store.list_nodes()}
    assert "NEW/Project/Alpha" in node_paths
    assert "NEW/Project/Beta" in node_paths
    assert all("OLD/Project" not in p for p in node_paths)


def test_rename_operation_via_executor_json(tmp_path):
    store = JsonStore(tmp_path)
    _populate(store, "OLD/Project")
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(
        operation="rename_namespace",
        target_path="OLD/Project",
        new_path="NEW/Project",
    ))
    paths = {c.node_path for c in store.list_chunks()}
    assert any("NEW/Project" in p for p in paths)
    assert all("OLD/Project" not in p for p in paths)


# ── SQLiteStore ───────────────────────────────────────────────────────────────

def test_rename_cascade_moves_chunks_sqlite(tmp_path):
    store = SQLiteStore(tmp_path)
    _populate(store, "OLD/Project")
    store.rename_namespace("OLD/Project", "NEW/Project")
    paths = {c.node_path for c in store.list_chunks()}
    assert all("OLD" not in p for p in paths)
    assert any("NEW/Project/Alpha" in p for p in paths)


def test_rename_cascade_moves_links_sqlite(tmp_path):
    store = SQLiteStore(tmp_path)
    _populate(store, "OLD/Project")
    store.rename_namespace("OLD/Project", "NEW/Project")
    for link in store.list_links():
        assert "OLD" not in link.source_path
        assert "OLD" not in link.target_path


def test_rename_operation_via_executor_sqlite(tmp_path):
    store = SQLiteStore(tmp_path)
    _populate(store, "OLD/Project")
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(
        operation="rename_namespace",
        target_path="OLD/Project",
        new_path="NEW/Project",
    ))
    paths = {c.node_path for c in store.list_chunks()}
    assert any("NEW/Project" in p for p in paths)
    assert all("OLD/Project" not in p for p in paths)


# ── Validation ────────────────────────────────────────────────────────────────

def test_rename_validation_requires_new_path(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    op = StorageOperation(operation="rename_namespace", target_path="OLD/Project", new_path=None)
    with pytest.raises(ValueError, match="new_path"):
        executor.apply(op)


def test_rename_validation_rejects_same_path(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    op = StorageOperation(
        operation="rename_namespace", target_path="OLD/Project", new_path="OLD/Project"
    )
    with pytest.raises(ValueError, match="new_path"):
        executor.apply(op)
