import json

from vertical_brain.core.models import ChunkInput, StorageOperation, StorageOperationBatch
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.storage.json_store import JsonStore
from vertical_brain.storage.sqlite_store import SQLiteStore


def _append_op(path="WORK/A/B", content="audited fact"):
    return StorageOperation(
        operation="append_chunk",
        target_path=path,
        chunk=ChunkInput(content=content),
        reasoning_summary="why this happened",
    )


def test_apply_writes_audit_record_sqlite(tmp_path):
    store = SQLiteStore(tmp_path)
    StorageOperationExecutor(store).apply(_append_op())
    audit = store.list_audit()
    assert len(audit) == 1
    assert audit[0]["operation_type"] == "append_chunk"


def test_apply_writes_audit_record_json(tmp_path):
    store = JsonStore(tmp_path)
    StorageOperationExecutor(store).apply(_append_op())
    audit = store.list_audit()
    assert len(audit) == 1
    assert audit[0]["operation_type"] == "append_chunk"


def test_dry_run_does_not_write_audit(tmp_path):
    store = SQLiteStore(tmp_path)
    StorageOperationExecutor(store).dry_run(_append_op())
    assert store.list_audit() == []

    json_store = JsonStore(tmp_path / "json")
    StorageOperationExecutor(json_store).dry_run(_append_op())
    assert json_store.list_audit() == []


def test_audit_record_contains_operation_details(tmp_path):
    store = JsonStore(tmp_path)
    StorageOperationExecutor(store).apply(_append_op())
    record = store.list_audit()[0]
    assert record["target_path"] == "WORK/A/B"
    assert record["status"] == "applied"
    assert record["reasoning_summary"] == "why this happened"
    payload = json.loads(record["payload_json"])
    assert payload["operation"] == "append_chunk"
    result = json.loads(record["result_json"])
    assert result["operation"] == "append_chunk"
    assert record["id"]
    assert record["created_at"]


def test_batch_apply_writes_audit_for_each_operation(tmp_path):
    store = SQLiteStore(tmp_path)
    batch = StorageOperationBatch(
        operations=[
            _append_op(content="one"),
            _append_op(content="two"),
            _append_op(content="three"),
        ]
    )
    StorageOperationExecutor(store).apply_batch(batch)
    assert len(store.list_audit()) == 3
