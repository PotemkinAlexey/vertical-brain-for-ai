"""Namespace rename/move cascade tests."""
from __future__ import annotations

import pytest

from vertical_brain.core.models import LinkInput, StorageOperation
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.storage.json_store import JsonStore
from vertical_brain.storage.sqlite_store import SQLiteStore


def _populate(store, prefix: str) -> None:
    from vertical_brain.core.models import ChunkInput
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


# ── SQLite atomicity ─────────────────────────────────────────────────────────

def test_rename_sqlite_uses_transaction_context(tmp_path):
    """rename_namespace must execute inside a transaction (all-or-nothing guarantee).

    We verify this by asserting the transaction depth is non-zero during the
    rename, using a subclass hook on the transaction() context manager.
    """
    import inspect
    source = inspect.getsource(SQLiteStore.rename_namespace)
    # The method must contain 'with self.transaction()' — any whitespace variant.
    assert "with self.transaction()" in source.replace("\n", " "), (
        "rename_namespace must wrap its UPDATEs in 'with self.transaction()'"
    )


def test_rename_sqlite_rolled_back_on_failure(tmp_path):
    """If the transaction context raises, all UPDATEs must be rolled back."""
    from contextlib import contextmanager
    store = SQLiteStore(tmp_path)
    _populate(store, "OLD/Project")
    original_chunks = frozenset(c.node_path for c in store.list_chunks())

    _real_transaction = store.transaction

    @contextmanager
    def _failing_transaction():
        with _real_transaction():
            yield
            raise RuntimeError("simulated failure after all UPDATEs")

    store.transaction = _failing_transaction  # type: ignore[method-assign]
    try:
        store.rename_namespace("OLD/Project", "NEW/Project")
    except RuntimeError:
        pass

    after = frozenset(c.node_path for c in store.list_chunks())
    assert after == original_chunks


# ── Validation ────────────────────────────────────────────────────────────────

def test_rename_validation_requires_new_path(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    op = StorageOperation(operation="rename_namespace", target_path="OLD/Project", new_path=None)
    with pytest.raises(ValueError, match="new_path"):
        executor.apply(op)


def test_rename_validation_rejects_same_path(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("OLD/Project")
    executor = StorageOperationExecutor(store)
    op = StorageOperation(
        operation="rename_namespace", target_path="OLD/Project", new_path="OLD/Project"
    )
    with pytest.raises(ValueError, match="new_path"):
        executor.apply(op)


def test_rename_validation_rejects_nonexistent_target(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    op = StorageOperation(
        operation="rename_namespace", target_path="GHOST/Project", new_path="NEW/Project"
    )
    with pytest.raises(ValueError, match="does not exist"):
        executor.apply(op)


def test_rename_validation_rejects_collision_with_existing_path(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("OLD/Project")
    store.ensure_node("NEW/Project")
    executor = StorageOperationExecutor(store)
    op = StorageOperation(
        operation="rename_namespace", target_path="OLD/Project", new_path="NEW/Project"
    )
    with pytest.raises(ValueError, match="already exists"):
        executor.apply(op)


def test_rename_validation_rejects_descendant_target(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/Project")
    executor = StorageOperationExecutor(store)
    op = StorageOperation(
        operation="rename_namespace",
        target_path="WORK/Project",
        new_path="WORK/Project/Sub",
    )
    with pytest.raises(ValueError, match="descendant"):
        executor.apply(op)


def test_rename_updates_node_name(tmp_path):
    store = JsonStore(tmp_path)
    _populate(store, "OLD/Project")
    StorageOperationExecutor(store).apply(StorageOperation(
        operation="rename_namespace", target_path="OLD/Project", new_path="NEW/Project"
    ))
    for node in store.list_nodes():
        if node.path.startswith("NEW/Project"):
            assert node.name == node.path.split("/")[-1], (
                f"node.name '{node.name}' does not match last segment of '{node.path}'"
            )


def test_rename_updates_parent_path(tmp_path):
    store = JsonStore(tmp_path)
    _populate(store, "OLD/Project")
    StorageOperationExecutor(store).apply(StorageOperation(
        operation="rename_namespace", target_path="OLD/Project", new_path="NEW/Project"
    ))
    for node in store.list_nodes():
        if node.path.startswith("NEW/Project/"):
            expected_parent = "/".join(node.path.split("/")[:-1])
            assert node.parent_path == expected_parent, (
                f"node '{node.path}': parent_path is '{node.parent_path}', expected '{expected_parent}'"
            )


def test_rename_sqlite_name_field_updated(tmp_path):
    store = SQLiteStore(tmp_path)
    _populate(store, "OLD/Project")
    store.rename_namespace("OLD/Project", "NEW/Project")
    for node in store.list_nodes():
        if node.path.startswith("NEW/Project"):
            assert node.name == node.path.split("/")[-1]


def test_rename_sqlite_search_works_after_rename(tmp_path):
    store = SQLiteStore(tmp_path)
    _populate(store, "OLD/Project")
    store.rename_namespace("OLD/Project", "NEW/Project")
    results = store.search("alpha")
    paths = {r.path for r in results}
    assert any("NEW/Project" in p for p in paths)
