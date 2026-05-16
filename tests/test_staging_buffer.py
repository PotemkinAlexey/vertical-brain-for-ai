"""STAGING buffer and valid_to auto-set tests."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from vertical_brain.core.models import STAGING_PATH, Chunk, ChunkInput, StorageOperation
from vertical_brain.core.operations import StorageOperationExecutor, operation_to_staging
from vertical_brain.storage.json_store import JsonStore


# ── operation_to_staging helper ───────────────────────────────────────────────

def test_operation_to_staging_targets_staging_path():
    op = operation_to_staging("some idea", original_target="WORK/DataArt", reason="low confidence")
    assert op.target_path == STAGING_PATH


def test_operation_to_staging_sets_source_staging():
    op = operation_to_staging("some idea", original_target="WORK/DataArt", reason="low confidence")
    assert op.chunk is not None
    assert op.chunk.source == "staging"


def test_operation_to_staging_sets_confidence_zero():
    op = operation_to_staging("some idea", original_target="WORK/DataArt", reason="low confidence")
    assert op.chunk is not None
    assert op.chunk.confidence == 0.0


def test_operation_to_staging_records_original_target_in_reasoning():
    op = operation_to_staging("some idea", original_target="WORK/DataArt", reason="conf=0.1")
    assert "WORK/DataArt" in op.reasoning_summary


def test_operation_to_staging_writes_bronze_chunk(tmp_path):
    store = JsonStore(tmp_path)
    op = operation_to_staging("unclassified note", original_target="WORK/DataArt", reason="x")
    StorageOperationExecutor(store).apply(op)

    chunks = store.get_chunks_by_path(STAGING_PATH)
    assert len(chunks) == 1
    assert chunks[0].content == "unclassified note"
    assert chunks[0].layer == "bronze"


# ── STAGING_PATH constant ─────────────────────────────────────────────────────

def test_staging_path_constant_value():
    assert STAGING_PATH == "STAGING/Unclassified"


# ── Doctor stale staging check ────────────────────────────────────────────────

def test_doctor_warns_on_old_staging_chunk(tmp_path):
    from datetime import timedelta
    from vertical_brain.core.doctor import Doctor
    store = JsonStore(tmp_path)
    chunk = Chunk(node_path=STAGING_PATH, content="forgotten note")
    chunk.created_at = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    store.save_chunk(chunk)

    issues = Doctor(store).run()
    assert any(i.check == "stale_staging_chunk" for i in issues)
    assert any(i.path == STAGING_PATH for i in issues)


def test_doctor_no_warning_for_fresh_staging_chunk(tmp_path):
    from vertical_brain.core.doctor import Doctor
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path=STAGING_PATH, content="fresh note"))

    issues = Doctor(store).run()
    assert not any(i.check == "stale_staging_chunk" for i in issues)


def test_doctor_no_warning_for_inactive_staging_chunk(tmp_path):
    from datetime import timedelta
    from vertical_brain.core.doctor import Doctor
    store = JsonStore(tmp_path)
    chunk = Chunk(node_path=STAGING_PATH, content="stale note", status="stale")
    chunk.created_at = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    store.save_chunk(chunk)

    issues = Doctor(store).run()
    assert not any(i.check == "stale_staging_chunk" for i in issues)


# ── valid_to auto-set on status transition ────────────────────────────────────

def test_mark_stale_sets_valid_to(tmp_path):
    store = JsonStore(tmp_path)
    chunk = store.save_chunk(Chunk(node_path="WORK/A", content="old fact"))
    assert chunk.valid_to is None

    StorageOperationExecutor(store).apply(StorageOperation(
        operation="mark_stale", target_path="WORK/A", chunk_ids=[chunk.id]
    ))

    updated = store.get_chunks_by_path("WORK/A")[0]
    assert updated.valid_to is not None
    datetime.fromisoformat(updated.valid_to)  # must be parseable ISO datetime


def test_supersede_chunk_sets_valid_to(tmp_path):
    store = JsonStore(tmp_path)
    chunk = store.save_chunk(Chunk(node_path="WORK/A", content="old fact"))

    StorageOperationExecutor(store).apply(StorageOperation(
        operation="supersede_chunk", target_path="WORK/A", chunk_ids=[chunk.id]
    ))

    updated = store.get_chunks_by_path("WORK/A")[0]
    assert updated.valid_to is not None


def test_active_chunk_has_no_valid_to(tmp_path):
    store = JsonStore(tmp_path)
    chunk = store.save_chunk(Chunk(node_path="WORK/A", content="current fact"))
    assert chunk.valid_to is None
