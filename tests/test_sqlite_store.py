import json

import pytest

from vertical_brain.core.models import Chunk, ChunkInput, Link, StorageOperation, StorageOperationBatch
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.storage.sqlite_store import SQLiteStore


def test_sqlite_store_creates_ancestor_chain(tmp_path):
    store = SQLiteStore(tmp_path)

    store.ensure_node("WORK/DataArt/Databricks")

    nodes = {node.path: node for node in store.list_nodes()}
    assert set(nodes) == {"WORK", "WORK/DataArt", "WORK/DataArt/Databricks"}
    assert nodes["WORK"].parent_path is None
    assert nodes["WORK/DataArt"].parent_path == "WORK"
    assert nodes["WORK/DataArt/Databricks"].parent_path == "WORK/DataArt"


def test_sqlite_store_seeds_configured_root_namespaces(tmp_path):
    namespace_dir = tmp_path / "namespaces"
    namespace_dir.mkdir()
    (namespace_dir / "root.json").write_text(
        json.dumps({"roots": ["WORK", "PERSONAL", "TRADING", "INBOX"]}),
        encoding="utf-8",
    )

    store = SQLiteStore(tmp_path)

    assert [node.path for node in store.list_nodes()] == ["WORK", "PERSONAL", "TRADING", "INBOX"]


def test_sqlite_store_persists_chunks_links_and_gold_summary(tmp_path):
    store = SQLiteStore(tmp_path)
    chunk = store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/Databricks",
            content="Delta fact",
            layer="silver",
            content_type="fact",
            lineage=["source-a", "source-b"],
        )
    )
    link = store.save_link(
        Link(
            source_path="WORK/DataArt/Databricks",
            target_path="WORK/DataArt/Databricks/AutoLoader",
            link_type="peer",
            reason="Related Databricks ingestion concepts.",
        )
    )
    repeated = store.save_link(
        Link(
            source_path=link.source_path,
            target_path=link.target_path,
            link_type=link.link_type,
            reason="Repeated route decision.",
        )
    )
    node = store.append_node_gold_aspect("WORK/DataArt/Databricks", "Stable summary")

    stored_chunk = store.get_chunks_by_path("WORK/DataArt/Databricks")[0]
    assert stored_chunk.id == chunk.id
    assert stored_chunk.lineage == ["source-a", "source-b"]
    assert repeated.id == link.id
    assert store.get_peer_paths("WORK/DataArt/Databricks") == ["WORK/DataArt/Databricks/AutoLoader"]
    assert node.gold_aspects == ["Stable summary"]
    assert store.gold_summary_path("WORK/DataArt/Databricks").read_text(encoding="utf-8") == (
        "# WORK/DataArt/Databricks\n\n- Stable summary\n"
    )


def test_sqlite_store_transaction_rolls_back_storage_writes(tmp_path):
    store = SQLiteStore(tmp_path)

    with pytest.raises(RuntimeError, match="rollback"):
        with store.transaction():
            store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Temporary fact"))
            raise RuntimeError("rollback")

    assert store.list_nodes() == []
    assert store.list_chunks() == []


def test_sqlite_operation_batch_runs_inside_transaction(tmp_path, monkeypatch):
    store = SQLiteStore(tmp_path)
    executor = StorageOperationExecutor(store)
    batch = StorageOperationBatch(
        operations=[
            StorageOperation(
                operation="append_chunk",
                target_path="WORK/Vertical/Node",
                chunk=ChunkInput(content="First write."),
            ),
            StorageOperation(
                operation="append_chunk",
                target_path="WORK/Vertical/Node",
                chunk=ChunkInput(content="Second write."),
            ),
        ]
    )
    original_save_chunk = store.save_chunk
    call_count = 0

    def failing_save_chunk(chunk):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated write failure")
        return original_save_chunk(chunk)

    monkeypatch.setattr(store, "save_chunk", failing_save_chunk)

    with pytest.raises(RuntimeError, match="simulated write failure"):
        executor.apply_batch(batch)

    assert store.get_chunks_by_path("WORK/Vertical/Node") == []
