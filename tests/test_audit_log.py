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


# ── MCP audit quality ────────────────────────────────────────────────────────

def _mcp_sqlite(tmp_path):
    from vertical_brain.mcp.server import VerticalBrainMCP
    store = SQLiteStore(tmp_path)
    return VerticalBrainMCP(store), store


def _call(mcp, name, args=None):
    import json
    from vertical_brain.mcp.server import VerticalBrainMCP
    return mcp.handle({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args or {}},
    })


def test_mcp_append_chunk_audit_has_default_summary(tmp_path):
    mcp, store = _mcp_sqlite(tmp_path)
    _call(mcp, "append_chunk", {"path": "WORK/A", "content": "a fact"})
    record = store.list_audit()[0]
    assert record["reasoning_summary"] == "Appended via MCP append_chunk."


def test_mcp_append_chunk_audit_uses_provided_summary(tmp_path):
    mcp, store = _mcp_sqlite(tmp_path)
    _call(mcp, "append_chunk", {
        "path": "WORK/A", "content": "a fact",
        "reasoning_summary": "explicit caller reason",
    })
    assert store.list_audit()[0]["reasoning_summary"] == "explicit caller reason"


def test_mcp_append_gold_aspect_audit_has_default_summary(tmp_path):
    mcp, store = _mcp_sqlite(tmp_path)
    _call(mcp, "append_gold_aspect", {"path": "WORK/A", "aspect": "gold label"})
    record = store.list_audit()[0]
    assert record["reasoning_summary"] == "Updated Gold aspect via MCP."


def test_mcp_create_link_audit_has_default_summary(tmp_path):
    mcp, store = _mcp_sqlite(tmp_path)
    _call(mcp, "create_link", {
        "source_path": "WORK/A", "target_path": "WORK/B",
        "link_type": "peer", "reason": "related",
    })
    record = store.list_audit()[0]
    assert record["reasoning_summary"] == "Created link via MCP."


def test_mcp_session_end_audit_has_default_summary(tmp_path):
    mcp, store = _mcp_sqlite(tmp_path)
    _call(mcp, "session_end", {"path": "WORK/A", "summary": "session wrap-up"})
    record = store.list_audit()[0]
    assert record["reasoning_summary"] == "Persisted session summary."


def test_mcp_mark_stale_without_reason_uses_default_summary(tmp_path):
    from vertical_brain.core.models import Chunk
    mcp, store = _mcp_sqlite(tmp_path)
    chunk = store.save_chunk(Chunk(node_path="WORK/A", content="old fact"))
    _call(mcp, "mark_stale", {"path": "WORK/A", "chunk_ids": [chunk.id]})
    record = store.list_audit()[0]
    assert record["reasoning_summary"] == "Marked stale via MCP."
    assert record["reasoning_summary"] != ""
