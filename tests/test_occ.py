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
