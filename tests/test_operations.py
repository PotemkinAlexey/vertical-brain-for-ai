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


def test_append_silver_first_silver_allowed(tmp_path):
    """append_chunk(layer=silver) is allowed when no active Silver exists yet."""
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    result = executor.apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK/DataArt",
        chunk=ChunkInput(content="summary v1", layer="silver"),
    ))
    assert result.status == "applied"
    chunks = store.get_chunks_by_path("WORK/DataArt")
    assert len(chunks) == 1
    assert chunks[0].layer == "silver"


def test_append_silver_second_silver_rejected(tmp_path):
    """append_chunk(layer=silver) is rejected when an active Silver already exists."""
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK/DataArt",
        chunk=ChunkInput(content="summary v1", layer="silver"),
    ))
    with pytest.raises(ValueError, match="update_silver"):
        executor.apply(StorageOperation(
            operation="append_chunk",
            target_path="WORK/DataArt",
            chunk=ChunkInput(content="summary v1", layer="silver"),
        ))


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


# ── update_silver ─────────────────────────────────────────────────────────────

def test_update_silver_replaces_active_silver(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    old = store.save_chunk(Chunk(node_path="WORK/A", content="old summary", layer="silver"))
    bronze = store.save_chunk(Chunk(node_path="WORK/A", content="new fact", layer="bronze"))

    result = executor.apply(StorageOperation(
        operation="update_silver",
        target_path="WORK/A",
        current_silver_id=old.id,
        source_chunk_ids=[bronze.id],
        chunk=ChunkInput(content="new summary", layer="silver"),
    ))

    chunks = store.get_chunks_by_path("WORK/A")
    old_chunk = next(c for c in chunks if c.id == old.id)
    new_chunk = next(c for c in chunks if c.id == result.chunk_id)

    assert old_chunk.status == "superseded"
    assert old_chunk.valid_to is not None
    assert new_chunk.status == "active"
    assert new_chunk.layer == "silver"
    assert old.id in new_chunk.lineage
    assert bronze.id in new_chunk.lineage
    assert old.id in new_chunk.supersedes


def test_update_silver_exactly_one_active_silver_after(tmp_path):
    """After update_silver there must be exactly one active Silver chunk."""
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    old = store.save_chunk(Chunk(node_path="WORK/A", content="v1", layer="silver"))

    result = executor.apply(StorageOperation(
        operation="update_silver",
        target_path="WORK/A",
        current_silver_id=old.id,
        chunk=ChunkInput(content="v2", layer="silver"),
    ))

    active_silver = [
        c for c in store.get_chunks_by_path("WORK/A")
        if c.layer == "silver" and c.status == "active"
    ]
    assert len(active_silver) == 1
    assert active_silver[0].id == result.chunk_id


def test_update_silver_chain_keeps_single_active_silver(tmp_path):
    """Two successive update_silver calls must leave exactly one active Silver."""
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    v1 = store.save_chunk(Chunk(node_path="WORK/A", content="v1", layer="silver"))

    r1 = executor.apply(StorageOperation(
        operation="update_silver", target_path="WORK/A",
        current_silver_id=v1.id, chunk=ChunkInput(content="v2", layer="silver"),
    ))
    r2 = executor.apply(StorageOperation(
        operation="update_silver", target_path="WORK/A",
        current_silver_id=r1.chunk_id, chunk=ChunkInput(content="v3", layer="silver"),
    ))

    active_silver = [
        c for c in store.get_chunks_by_path("WORK/A")
        if c.layer == "silver" and c.status == "active"
    ]
    assert len(active_silver) == 1
    assert active_silver[0].id == r2.chunk_id


def test_update_silver_rejects_missing_current_silver_id(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    validation = executor.validate(StorageOperation(
        operation="update_silver", target_path="WORK/A",
        current_silver_id=None,
        chunk=ChunkInput(content="new summary", layer="silver"),
    ))
    assert not validation.valid
    assert any("current_silver_id" in i.path for i in validation.issues)


def test_update_silver_rejects_id_from_another_namespace(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    other = store.save_chunk(Chunk(node_path="WORK/B", content="other Silver", layer="silver"))
    with pytest.raises(ValueError, match="belongs to"):
        executor.apply(StorageOperation(
            operation="update_silver", target_path="WORK/A",
            current_silver_id=other.id,
            chunk=ChunkInput(content="new summary", layer="silver"),
        ))


def test_update_silver_rejects_bronze_as_current_silver_id(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    bronze = store.save_chunk(Chunk(node_path="WORK/A", content="a fact", layer="bronze"))
    with pytest.raises(ValueError, match="layer=bronze"):
        executor.apply(StorageOperation(
            operation="update_silver", target_path="WORK/A",
            current_silver_id=bronze.id,
            chunk=ChunkInput(content="summary", layer="silver"),
        ))


def test_update_silver_rejects_superseded_current_silver_id(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    old = store.save_chunk(Chunk(node_path="WORK/A", content="v1", layer="silver"))
    old.status = "superseded"
    store.update_chunk(old)
    with pytest.raises(ValueError, match="superseded"):
        executor.apply(StorageOperation(
            operation="update_silver", target_path="WORK/A",
            current_silver_id=old.id,
            chunk=ChunkInput(content="v2", layer="silver"),
        ))


def test_update_silver_rejects_gold_source_chunk(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    silver = store.save_chunk(Chunk(node_path="WORK/A", content="summary", layer="silver"))
    gold = store.save_chunk(Chunk(node_path="WORK/A", content="gold", layer="gold"))
    with pytest.raises(ValueError, match="Gold cannot be used"):
        executor.apply(StorageOperation(
            operation="update_silver", target_path="WORK/A",
            current_silver_id=silver.id,
            source_chunk_ids=[gold.id],
            chunk=ChunkInput(content="new summary", layer="silver"),
        ))


def test_update_silver_rejects_duplicate_source_chunk_ids(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    silver = store.save_chunk(Chunk(node_path="WORK/A", content="summary", layer="silver"))
    bronze = store.save_chunk(Chunk(node_path="WORK/A", content="fact", layer="bronze"))
    validation = executor.validate(StorageOperation(
        operation="update_silver", target_path="WORK/A",
        current_silver_id=silver.id,
        source_chunk_ids=[bronze.id, bronze.id],
        chunk=ChunkInput(content="new summary", layer="silver"),
    ))
    assert not validation.valid
    assert any("source_chunk_ids" in i.path for i in validation.issues)


def test_update_silver_increments_version_and_marks_dirty(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    store.ensure_node("WORK/A")
    silver = store.save_chunk(Chunk(node_path="WORK/A", content="v1", layer="silver"))
    v0 = store.get_node("WORK/A").version

    executor.apply(StorageOperation(
        operation="update_silver", target_path="WORK/A",
        current_silver_id=silver.id,
        chunk=ChunkInput(content="v2", layer="silver"),
    ))

    node = store.get_node("WORK/A")
    assert node.version > v0
    assert node.is_dirty is True


def test_append_chunk_silver_rejected_when_active_silver_exists(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    store.save_chunk(Chunk(node_path="WORK/A", content="existing Silver", layer="silver"))
    with pytest.raises(ValueError, match="update_silver"):
        executor.apply(StorageOperation(
            operation="append_chunk", target_path="WORK/A",
            chunk=ChunkInput(content="second Silver", layer="silver"),
        ))


def test_append_gold_aspect_succeeds_after_update_silver(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    old = store.save_chunk(Chunk(node_path="WORK/A", content="v1 silver", layer="silver"))
    executor.apply(StorageOperation(
        operation="update_silver", target_path="WORK/A",
        current_silver_id=old.id,
        chunk=ChunkInput(content="v2 silver", layer="silver"),
    ))
    # Must not raise — new active Silver exists
    executor.apply(StorageOperation(
        operation="append_gold_aspect", target_path="WORK/A",
        gold_aspect="stable conclusion",
    ))
    gold = [c for c in store.get_chunks_by_path("WORK/A") if c.layer == "gold" and c.status == "active"]
    assert len(gold) == 1
