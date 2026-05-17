import json
import sqlite3

import pytest

from vertical_brain.core.gold import GoldAspect, gold_aspect_embed_key, serialize_gold_aspects
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
    gold = store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Stable summary", layer="gold"))

    stored_chunk = store.get_chunks_by_path("WORK/DataArt/Databricks")[0]
    assert stored_chunk.id == chunk.id
    assert stored_chunk.lineage == ["source-a", "source-b"]
    assert repeated.id == link.id
    assert store.get_link(link.id) == link
    assert store.get_peer_paths("WORK/DataArt/Databricks") == ["WORK/DataArt/Databricks/AutoLoader"]
    gold_chunks = [c for c in store.get_chunks_by_path("WORK/DataArt/Databricks") if c.layer == "gold"]
    assert len(gold_chunks) == 1
    assert gold_chunks[0].id == gold.id


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


def test_sqlite_wal_and_synchronous_pragmas_are_set(tmp_path):
    """SQLiteStore must open with WAL + synchronous=NORMAL for concurrent safety."""
    store = SQLiteStore(tmp_path)
    journal = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
    synchronous = store.conn.execute("PRAGMA synchronous").fetchone()[0]
    assert journal == "wal"
    assert synchronous == 1  # 1 = NORMAL


def test_sqlite_backup_to_creates_readable_copy(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="durable fact"))

    backup_file = store.backup_to(tmp_path / "backups" / "brain.sqlite")
    backup = SQLiteStore(backup_file.parent, db_name=backup_file.name)

    chunks = backup.get_chunks_by_path("WORK/A")
    assert len(chunks) == 1
    assert chunks[0].content == "durable fact"


def test_sqlite_backup_refuses_overwrite_by_default(tmp_path):
    store = SQLiteStore(tmp_path)
    backup_file = tmp_path / "brain-backup.sqlite"
    store.backup_to(backup_file)

    with pytest.raises(FileExistsError, match="already exists"):
        store.backup_to(backup_file)

    store.backup_to(backup_file, overwrite=True)


def test_sqlite_backup_rejects_open_transaction(tmp_path):
    store = SQLiteStore(tmp_path)

    with store.transaction():
        with pytest.raises(RuntimeError, match="transaction"):
            store.backup_to(tmp_path / "backup.sqlite")


def test_sqlite_backup_rejects_source_database_path(tmp_path):
    store = SQLiteStore(tmp_path)

    with pytest.raises(ValueError, match="must differ"):
        store.backup_to(store.db_file, overwrite=True)


def test_sqlite_checkpoint_returns_wal_status(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="checkpoint fact"))

    result = store.checkpoint("passive")

    assert set(result) == {"busy", "log", "checkpointed"}
    assert all(isinstance(value, int) for value in result.values())


def test_sqlite_checkpoint_rejects_unknown_mode(tmp_path):
    store = SQLiteStore(tmp_path)

    with pytest.raises(ValueError, match="Unsupported checkpoint mode"):
        store.checkpoint("vacuum")


def test_sqlite_vacuum_dry_run_reports_inactive_chunks_without_mutating(tmp_path):
    store = SQLiteStore(tmp_path)
    stale = store.save_chunk(
        Chunk(
            node_path="WORK/Old",
            content="old fact",
            status="stale",
            valid_to="2000-01-01T00:00:00+00:00",
            updated_at="2000-01-01T00:00:00+00:00",
        )
    )
    store.set_vector(stale.content_hash, "mock", [1.0, 0.0])

    result = store.vacuum(retention_hours=0, dry_run=True)

    assert result["dry_run"] is True
    assert result["eligible_chunks"] == 1
    assert result["eligible_by_status"] == {"stale": 1}
    assert result["deleted_chunks"] == 0
    assert store.get_chunks_by_path("WORK/Old")[0].id == stale.id
    assert store.get_vector(stale.content_hash, "mock") == [1.0, 0.0]


def test_sqlite_vacuum_refuses_low_retention_apply_without_force(tmp_path):
    store = SQLiteStore(tmp_path)

    with pytest.raises(ValueError, match="requires force=True"):
        store.vacuum(retention_hours=0, dry_run=False)


def test_sqlite_vacuum_deletes_inactive_chunks_vectors_and_empty_nodes(tmp_path):
    store = SQLiteStore(tmp_path)
    active = store.save_chunk(Chunk(node_path="WORK/Keep", content="current fact"))
    stale = store.save_chunk(
        Chunk(
            node_path="WORK/Old/Leaf",
            content="obsolete searchable phrase",
            status="stale",
            valid_to="2000-01-01T00:00:00+00:00",
            updated_at="2000-01-01T00:00:00+00:00",
        )
    )
    store.set_vector(active.content_hash, "mock", [1.0, 0.0])
    store.set_vector(stale.content_hash, "mock", [0.0, 1.0])

    assert store.search("obsolete searchable phrase", include_stale=True)

    result = store.vacuum(retention_hours=0, dry_run=False, force=True)

    assert result["deleted_chunks"] == 1
    assert result["deleted_vectors"] == 1
    assert result["deleted_empty_nodes"] == 2
    assert result["checkpoint"] is not None
    assert store.get_chunks_by_path("WORK/Old/Leaf") == []
    assert store.get_node("WORK/Old") is None
    assert store.get_node("WORK/Old/Leaf") is None
    assert store.get_node("WORK/Keep") is not None
    assert store.get_vector(active.content_hash, "mock") == [1.0, 0.0]
    assert store.get_vector(stale.content_hash, "mock") is None
    assert store.search("obsolete searchable phrase", include_stale=True) == []


def test_sqlite_vacuum_keeps_active_gold_aspect_vectors(tmp_path):
    store = SQLiteStore(tmp_path)
    gold = store.save_chunk(Chunk(
        node_path="WORK/Gold",
        content=serialize_gold_aspects([GoldAspect(text="Delta Z-ordering")]),
        layer="gold",
    ))
    active_key = gold_aspect_embed_key("Delta Z-ordering")
    stale_key = gold_aspect_embed_key("Obsolete Gold tag")
    store.set_vector(gold.content_hash, "mock", [0.5])
    store.set_vector(active_key, "mock", [1.0, 0.0])
    store.set_vector(stale_key, "mock", [0.0, 1.0])

    result = store.vacuum(retention_hours=0, dry_run=False, force=True)

    assert result["deleted_vectors"] == 1
    assert store.get_vector(gold.content_hash, "mock") == [0.5]
    assert store.get_vector(active_key, "mock") == [1.0, 0.0]
    assert store.get_vector(stale_key, "mock") is None


def test_sqlite_vacuum_respects_retention_window(tmp_path):
    store = SQLiteStore(tmp_path)
    stale = store.save_chunk(
        Chunk(
            node_path="WORK/Recent",
            content="recent stale fact",
            status="stale",
        )
    )

    result = store.vacuum(retention_hours=168, dry_run=False)

    assert result["eligible_chunks"] == 0
    assert result["deleted_chunks"] == 0
    assert store.get_chunks_by_path("WORK/Recent")[0].id == stale.id


def test_sqlite_close_closes_connection(tmp_path):
    store = SQLiteStore(tmp_path)
    store.close()

    with pytest.raises(sqlite3.ProgrammingError):
        store.list_nodes()


def test_sqlite_fts_does_not_leak_superseded_chunks(tmp_path):
    """After supersede, the FTS index must not return the superseded chunk."""
    store = SQLiteStore(tmp_path)
    if not store._fts_enabled:
        pytest.skip("FTS not available")
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="unique telltale phrase"))
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="unrelated second fact"))

    chunks = store.get_chunks_by_path("WORK/DataArt/Databricks")
    telltale = next(c for c in chunks if "unique telltale phrase" in c.content)

    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(
        operation="supersede_chunk",
        target_path="WORK/DataArt/Databricks",
        chunk_ids=[telltale.id],
        reasoning_summary="Supersede for FTS test.",
    ))

    results = store.search("unique telltale phrase", limit=10)
    assert not any(r.chunk_id == telltale.id for r in results)


def test_vacuum_skips_immutable_chunks(tmp_path):
    store = SQLiteStore(tmp_path)
    # Write an immutable stale chunk directly (force status via update_chunk)
    chunk = Chunk(node_path="WORK/A", content="immutable spec", layer="bronze",
                  status="stale", immutable=True)
    store.ensure_node("WORK/A")
    store.save_chunk(chunk)
    result = store.vacuum(retention_hours=0, dry_run=False, force=True)
    # Immutable chunk must not appear in deleted count
    assert result["deleted_chunks"] == 0
    remaining = store.get_chunks_by_path("WORK/A", include_children=False)
    assert any(c.id == chunk.id for c in remaining)
