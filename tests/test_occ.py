"""Optimistic Concurrency Control (OCC) tests."""
from __future__ import annotations

import pytest

from vertical_brain.core.models import OptimisticLockException, StorageOperationBatch
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.core.optimizer import SimpleOptimizer
from vertical_brain.storage.json_store import JsonStore


def _make_batch_with_version(store: JsonStore, path: str) -> StorageOperationBatch:
    opt = SimpleOptimizer(store, min_compaction_path_parts=2)
    store.ensure_node(path)
    return opt.plan_branch(path)


def test_plan_branch_captures_start_version(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/Project")
    batch = SimpleOptimizer(store, min_compaction_path_parts=2).plan_branch("WORK/Project")
    assert batch.branch_path == "WORK/Project"
    assert batch.start_version == 0


def test_plan_branch_captures_version_before_chunk_snapshot_for_race_detection(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/Project")
    original_get_chunks = store.get_chunks_by_path

    def racing_get_chunks(path: str, include_children: bool = False):
        chunks = original_get_chunks(path, include_children=include_children)
        node = store.get_node("WORK/Project")
        assert node is not None
        node.version += 1
        store.update_node(node)
        return chunks

    store.get_chunks_by_path = racing_get_chunks  # type: ignore[method-assign]

    batch = SimpleOptimizer(store, min_compaction_path_parts=2).plan_branch("WORK/Project")

    assert batch.start_version == 0
    with pytest.raises(OptimisticLockException, match="version"):
        StorageOperationExecutor(store).apply_batch(batch)


def test_apply_batch_succeeds_when_version_matches(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/Project")
    batch = SimpleOptimizer(store, min_compaction_path_parts=2).plan_branch("WORK/Project")
    # No mutations → version still 0, batch should apply cleanly (even if empty).
    batch.operations = []  # force non-empty path to verify no exception
    executor = StorageOperationExecutor(store)
    result = executor.apply_batch(batch)
    assert result.status == "applied"


def test_apply_batch_raises_on_version_mismatch(tmp_path):
    from vertical_brain.core.models import ChunkInput, StorageOperation
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/Project")

    batch = SimpleOptimizer(store, min_compaction_path_parts=2).plan_branch("WORK/Project")
    batch.operations = []

    # Simulate a concurrent write that bumps the version.
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK/Project",
        chunk=ChunkInput(content="concurrent write"),
    ))

    # Now the batch's start_version (0) is stale.
    with pytest.raises(OptimisticLockException, match="version"):
        executor.apply_batch(batch)


def test_apply_batch_skips_occ_when_branch_path_is_none(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/Project")
    batch = StorageOperationBatch(operations=[], branch_path=None, start_version=None)
    result = StorageOperationExecutor(store).apply_batch(batch)
    assert result.status == "applied"


def test_occ_failure_writes_no_chunks_and_no_audit(tmp_path):
    """A failed OCC check must leave no partial writes and no audit records."""
    from vertical_brain.core.models import ChunkInput, StorageOperation
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/Project")

    batch = SimpleOptimizer(store, min_compaction_path_parts=2).plan_branch("WORK/Project")
    batch.operations = [
        StorageOperation(
            operation="append_chunk",
            target_path="WORK/Project",
            chunk=ChunkInput(content="should not land"),
        )
    ]

    # Bump version to cause conflict.
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK/Project",
        chunk=ChunkInput(content="concurrent write"),
    ))
    before_chunks = len(store.list_chunks())
    before_audit = len(store.list_audit())

    with pytest.raises(OptimisticLockException):
        executor.apply_batch(batch)

    assert len(store.list_chunks()) == before_chunks
    assert len(store.list_audit()) == before_audit


def test_occ_check_inside_transaction_for_sqlite(tmp_path):
    """For SQLiteStore the check and writes are inside the same transaction."""
    from vertical_brain.core.models import ChunkInput, StorageOperation
    from vertical_brain.storage.sqlite_store import SQLiteStore

    store = SQLiteStore(tmp_path)
    store.ensure_node("WORK/Project")

    batch = SimpleOptimizer(store, min_compaction_path_parts=2).plan_branch("WORK/Project")
    batch.operations = [
        StorageOperation(
            operation="append_chunk",
            target_path="WORK/Project",
            chunk=ChunkInput(content="should not land"),
        )
    ]

    # Advance version concurrently.
    StorageOperationExecutor(store).apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK/Project",
        chunk=ChunkInput(content="concurrent write"),
    ))
    chunk_count_before = len(store.list_chunks())

    with pytest.raises(OptimisticLockException, match="version"):
        StorageOperationExecutor(store).apply_batch(batch)

    # The "should not land" chunk must not be in the DB.
    assert len(store.list_chunks()) == chunk_count_before
