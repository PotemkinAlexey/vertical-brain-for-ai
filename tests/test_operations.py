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


def test_json_store_batch_rolls_back_when_apply_fails_mid_write(tmp_path):
    store = JsonStore(tmp_path)
    batch = StorageOperationBatch(
        operations=[
            StorageOperation(
                operation="append_chunk",
                target_path="WORK/Vertical/Node",
                chunk=ChunkInput(content="must be rolled back"),
            )
        ]
    )

    def failing_update_node(node):
        raise RuntimeError("simulated node update failure")

    store.update_node = failing_update_node  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="simulated node update failure"):
        StorageOperationExecutor(store).apply_batch(batch)

    assert store.list_chunks() == []
    assert store.list_nodes() == []
    assert store.list_audit() == []


def test_batch_validation_requires_complete_occ_pair(tmp_path):
    store = JsonStore(tmp_path)
    batch = StorageOperationBatch(
        operations=[],
        branch_path="WORK/Vertical",
        start_version=None,
    )

    result = StorageOperationExecutor(store).dry_run_batch(batch)

    assert result.status == "invalid"
    assert result.validation is not None
    assert any("provided together" in issue.message for issue in result.validation.issues)


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


def test_append_gold_aspect_rejected_without_silver(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    import pytest
    with pytest.raises(ValueError, match="no active Silver chunk found"):
        executor.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/DataArt", gold_aspect="orphan gold"))


def test_append_gold_aspect_creates_gold_chunk_and_extends_on_fit(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="silver summary", layer="silver"))

    executor.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/DataArt", gold_aspect="migration"))
    executor.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/DataArt", gold_aspect="clusters"))

    gold = [c for c in store.get_chunks_by_path("WORK/DataArt") if c.layer == "gold" and c.status == "active"]
    assert len(gold) == 1
    from vertical_brain.core.gold import parse_gold_content
    assert parse_gold_content(gold[0].content) == ["migration", "clusters"]


def test_append_gold_aspect_creates_overflow_sibling_when_full(tmp_path):
    from vertical_brain.core.gold import MAX_GOLD_ASPECTS, GoldAspect, parse_gold_content, serialize_gold_aspects
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="silver summary", layer="silver"))

    # Fill the primary node to MAX_GOLD_ASPECTS
    full_content = serialize_gold_aspects([GoldAspect(text=f"a{i}") for i in range(MAX_GOLD_ASPECTS)])
    store.save_chunk(Chunk(node_path="WORK/DataArt", content=full_content, layer="gold"))
    executor.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/DataArt", gold_aspect="overflow"))

    primary_gold = [c for c in store.get_chunks_by_path("WORK/DataArt") if c.layer == "gold" and c.status == "active"]
    assert len(primary_gold) == 1
    assert len(parse_gold_content(primary_gold[0].content)) == MAX_GOLD_ASPECTS

    overflow_links = [lnk for lnk in store.list_links() if lnk.link_type == "gold_overflow"]
    assert len(overflow_links) == 1
    overflow_path = overflow_links[0].target_path
    assert overflow_path == "WORK/DataArt_2"

    overflow_gold = [c for c in store.get_chunks_by_path(overflow_path) if c.layer == "gold" and c.status == "active"]
    assert len(overflow_gold) == 1
    assert parse_gold_content(overflow_gold[0].content) == ["overflow"]


# ── Gold dirty flag ───────────────────────────────────────────────────────────

def test_append_chunk_marks_ancestor_nodes_dirty(tmp_path):
    """append_chunk must set is_dirty=True on the target node and all ancestors."""
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK/DataArt/Databricks",
        chunk=ChunkInput(content="new fact about Databricks"),
        reasoning_summary="Dirty flag test.",
    ))

    nodes = {n.path: n for n in store.list_nodes()}
    assert nodes["WORK"].is_dirty is True
    assert nodes["WORK/DataArt"].is_dirty is True
    assert nodes["WORK/DataArt/Databricks"].is_dirty is True


def test_mark_stale_marks_ancestor_nodes_dirty(tmp_path):
    """mark_stale must propagate is_dirty up the hierarchy."""
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    chunk = store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="fact to stale"))
    # Manually clear dirty flags that were set by save_chunk path creation.
    for node in store.list_nodes():
        node.is_dirty = False
        store.update_node(node)

    executor.apply(StorageOperation(
        operation="mark_stale",
        target_path="WORK/DataArt/Databricks",
        chunk_ids=[chunk.id],
        reasoning_summary="Stale dirty flag test.",
    ))

    nodes = {n.path: n for n in store.list_nodes()}
    assert nodes["WORK/DataArt/Databricks"].is_dirty is True
    assert nodes["WORK/DataArt"].is_dirty is True


# ── Bronze dedup guard ───────────────────────────────────────────────────────

def test_append_bronze_duplicate_raises_error(tmp_path):
    """Writing the same Bronze content twice to the same path must be rejected."""
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK/DataArt",
        chunk=ChunkInput(content="Databricks uses Delta Lake.", layer="bronze"),
    ))

    with pytest.raises(ValueError, match="identical content already exists"):
        executor.apply(StorageOperation(
            operation="append_chunk",
            target_path="WORK/DataArt",
            chunk=ChunkInput(content="Databricks uses Delta Lake.", layer="bronze"),
        ))

    # Only one chunk must exist
    assert len(store.get_chunks_by_path("WORK/DataArt")) == 1


def test_append_bronze_duplicate_does_not_write_chunk(tmp_path):
    """Store must be unchanged after a rejected duplicate Bronze write."""
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK/DataArt",
        chunk=ChunkInput(content="stable fact"),
    ))

    try:
        executor.apply(StorageOperation(
            operation="append_chunk",
            target_path="WORK/DataArt",
            chunk=ChunkInput(content="stable fact"),
        ))
    except ValueError:
        pass

    assert len(store.get_chunks_by_path("WORK/DataArt")) == 1


def test_append_bronze_allows_different_content_same_path(tmp_path):
    """Two distinct Bronze facts at the same path must both be accepted."""
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK/DataArt",
        chunk=ChunkInput(content="fact one"),
    ))
    executor.apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK/DataArt",
        chunk=ChunkInput(content="fact two"),
    ))

    assert len(store.get_chunks_by_path("WORK/DataArt")) == 2


def test_append_silver_duplicate_is_allowed(tmp_path):
    """Dedup guard applies only to Bronze; Silver can be overwritten freely."""
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK/DataArt",
        chunk=ChunkInput(content="summary v1", layer="silver"),
    ))
    # Writing the same Silver content again must NOT raise.
    executor.apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK/DataArt",
        chunk=ChunkInput(content="summary v1", layer="silver"),
    ))

    assert len(store.get_chunks_by_path("WORK/DataArt")) == 2


def test_append_bronze_similar_content_returns_soft_warn(tmp_path):
    """Writing a near-duplicate Bronze chunk must succeed but populate similar_bronze."""
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK/DataArt",
        chunk=ChunkInput(content="Databricks uses Delta Lake for streaming ingestion."),
    ))

    result = executor.apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK/DataArt",
        chunk=ChunkInput(content="Databricks Delta Lake handles streaming ingestion jobs."),
    ))

    # Write succeeded
    assert result.status == "applied"
    assert result.chunk_id is not None
    # But similar_bronze points at the earlier chunk
    assert len(result.similar_bronze) >= 1
    assert result.similar_bronze[0].layer == "bronze"


def test_append_bronze_unrelated_content_no_similar_warn(tmp_path):
    """Writing a truly different Bronze fact must not produce any similar_bronze warning."""
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK/DataArt",
        chunk=ChunkInput(content="Databricks uses Delta Lake."),
    ))

    result = executor.apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK/DataArt",
        chunk=ChunkInput(content="Python pandas is used for data wrangling."),
    ))

    assert result.status == "applied"
    assert result.similar_bronze == []


def test_append_bronze_duplicate_error_message_hints_mark_stale(tmp_path):
    """The rejection error message must guide the agent toward mark_stale."""
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK/DataArt",
        chunk=ChunkInput(content="old fact"),
    ))

    with pytest.raises(ValueError, match="mark_stale"):
        executor.apply(StorageOperation(
            operation="append_chunk",
            target_path="WORK/DataArt",
            chunk=ChunkInput(content="old fact"),
        ))


def test_dirty_flag_only_set_for_mutations_not_reads(tmp_path):
    """Simply reading chunks must not change is_dirty on any node."""
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="read-only fact"))
    for node in store.list_nodes():
        node.is_dirty = False
        store.update_node(node)

    _ = store.get_chunks_by_path("WORK/DataArt/Databricks")

    nodes = {n.path: n for n in store.list_nodes()}
    assert all(not n.is_dirty for n in nodes.values())
