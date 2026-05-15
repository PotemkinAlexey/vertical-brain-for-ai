import pytest

from vertical_brain.core.models import (
    Chunk,
    ChunkInput,
    LinkInput,
    StaleCandidateInput,
    StorageOperation,
    StorageOperationBatch,
)
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.storage.json_store import JsonStore


def test_append_chunk_operation_writes_chunk_links_and_stale_candidates(tmp_path):
    store = JsonStore(tmp_path)
    operation = StorageOperation(
        operation="append_chunk",
        target_path="WORK/Vertical/Node",
        chunk=ChunkInput(
            content="A durable fact.",
            layer="silver",
            content_type="fact",
            confidence=0.91,
        ),
        links=[
            LinkInput(
                target_path="WORK/Other/Node",
                link_type="peer",
                reason="Explicit horizontal link.",
            )
        ],
        stale_candidates=[
            StaleCandidateInput(
                path="WORK/Vertical/Node",
                reason="Model identified a possible stale chunk.",
            )
        ],
        confidence=0.91,
        reasoning_summary="Model requested append_chunk.",
    )

    result = StorageOperationExecutor(store).apply(operation)

    chunks = store.get_chunks_by_path("WORK/Vertical/Node")
    assert result.operation == "append_chunk"
    assert result.chunk_id == chunks[0].id
    assert chunks[0].content == "A durable fact."
    assert chunks[0].source == "model"
    assert store.get_peer_paths("WORK/Vertical/Node") == ["WORK/Other/Node"]
    assert result.stale_candidates[0].path == "WORK/Vertical/Node"


def test_create_node_operation_creates_ancestor_chain(tmp_path):
    store = JsonStore(tmp_path)
    operation = StorageOperation(operation="create_node", target_path="WORK/New/Branch")

    result = StorageOperationExecutor(store).apply(operation)

    assert result.status == "applied"
    assert store.get_node("WORK/New/Branch") is not None


def test_operation_batch_appends_canonical_chunk_and_supersedes_variants(tmp_path):
    store = JsonStore(tmp_path)
    path = "WORK/Vertical/Node/Topic/Variant"
    first = store.save_chunk(Chunk(node_path=path, content="Variant A"))
    second = store.save_chunk(Chunk(node_path=path, content="Variant B"))

    batch = StorageOperationBatch(
        operations=[
            StorageOperation(
                operation="append_chunk",
                target_path=path,
                chunk=ChunkInput(
                    content="Canonical variant.",
                    layer="silver",
                    content_type="fact",
                    source="optimizer:namespace_compaction",
                    lineage=[first.id, second.id],
                ),
            ),
            StorageOperation(
                operation="supersede_chunk",
                target_path=path,
                chunk_ids=[first.id, second.id],
            ),
        ],
        reasoning_summary="Collapse variants into one canonical chunk.",
    )

    result = StorageOperationExecutor(store).apply_batch(batch)

    chunks = store.get_chunks_by_path(path)
    canonical = [chunk for chunk in chunks if chunk.source == "optimizer:namespace_compaction"]
    originals = [chunk for chunk in chunks if chunk.id in {first.id, second.id}]
    assert result.status == "applied"
    assert len(result.results) == 2
    assert len(canonical) == 1
    assert canonical[0].lineage == [first.id, second.id]
    assert {chunk.status for chunk in originals} == {"superseded"}


def test_dry_run_reports_valid_operation_without_mutating_store(tmp_path):
    store = JsonStore(tmp_path)
    operation = StorageOperation(
        operation="append_chunk",
        target_path="WORK/Vertical/Node",
        chunk=ChunkInput(content="Dry-run only."),
    )

    result = StorageOperationExecutor(store).dry_run(operation)

    assert result.status == "dry_run"
    assert result.validation is not None
    assert result.validation.valid is True
    assert store.get_chunks_by_path("WORK/Vertical/Node") == []


def test_validation_rejects_invalid_append_chunk_without_mutating_store(tmp_path):
    store = JsonStore(tmp_path)
    operation = StorageOperation(
        operation="append_chunk",
        target_path="WORK/Vertical/Node",
        chunk=ChunkInput(content="", layer="unknown"),
    )
    executor = StorageOperationExecutor(store)

    validation = executor.validate(operation)

    assert validation.valid is False
    assert {issue.path for issue in validation.issues} == {
        "operation.chunk.content",
        "operation.chunk.layer",
    }
    with pytest.raises(ValueError, match="chunk content must be non-empty"):
        executor.apply(operation)
    assert store.get_chunks_by_path("WORK/Vertical/Node") == []


def test_batch_prevalidation_prevents_partial_mutation(tmp_path):
    store = JsonStore(tmp_path)
    existing = store.save_chunk(Chunk(node_path="WORK/Other/Node", content="Existing chunk."))
    batch = StorageOperationBatch(
        operations=[
            StorageOperation(
                operation="append_chunk",
                target_path="WORK/Vertical/Node",
                chunk=ChunkInput(content="Should not be written."),
            ),
            StorageOperation(
                operation="supersede_chunk",
                target_path="WORK/Vertical/Node",
                chunk_ids=[existing.id],
            ),
        ]
    )

    with pytest.raises(ValueError, match="does not belong"):
        StorageOperationExecutor(store).apply_batch(batch)

    assert store.get_chunks_by_path("WORK/Vertical/Node") == []
    assert store.get_chunks_by_path("WORK/Other/Node")[0].status == "active"


def test_dry_run_batch_returns_invalid_result_without_mutating_store(tmp_path):
    store = JsonStore(tmp_path)
    batch = StorageOperationBatch(
        operations=[
            StorageOperation(
                operation="append_chunk",
                target_path="WORK/Vertical/Node",
                chunk=ChunkInput(content="Valid first operation."),
            ),
            StorageOperation(
                operation="create_link",
                target_path="WORK/Vertical/Node",
                links=[],
            ),
        ]
    )

    result = StorageOperationExecutor(store).dry_run_batch(batch)

    assert result.status == "invalid"
    assert result.validation is not None
    assert result.validation.valid is False
    assert result.results == []
    assert store.get_chunks_by_path("WORK/Vertical/Node") == []


def test_append_gold_aspect_creates_gold_chunk_and_extends_on_fit(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)

    executor.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/DataArt", gold_aspect="migration"))
    executor.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/DataArt", gold_aspect="clusters"))

    gold = [c for c in store.get_chunks_by_path("WORK/DataArt") if c.layer == "gold" and c.status == "active"]
    assert len(gold) == 1
    assert gold[0].content == "migration | clusters"


def test_append_gold_aspect_creates_overflow_sibling_when_full(tmp_path):
    from vertical_brain.core.operations import MAX_GOLD_CHARS
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)

    long_aspect = "x" * (MAX_GOLD_CHARS - 5)
    executor.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/DataArt", gold_aspect=long_aspect))
    executor.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/DataArt", gold_aspect="overflow"))

    primary_gold = [c for c in store.get_chunks_by_path("WORK/DataArt") if c.layer == "gold" and c.status == "active"]
    assert len(primary_gold) == 1
    assert primary_gold[0].content == long_aspect

    overflow_links = [lnk for lnk in store.list_links() if lnk.link_type == "gold_overflow"]
    assert len(overflow_links) == 1
    overflow_path = overflow_links[0].target_path
    assert overflow_path == "WORK/DataArt_2"

    overflow_gold = [c for c in store.get_chunks_by_path(overflow_path) if c.layer == "gold" and c.status == "active"]
    assert len(overflow_gold) == 1
    assert overflow_gold[0].content == "overflow"
