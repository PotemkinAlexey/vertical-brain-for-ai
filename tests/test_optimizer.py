from unittest.mock import MagicMock, patch

from vertical_brain.core.models import Chunk
from vertical_brain.core.optimizer import SimpleOptimizer
from vertical_brain.llm.embedding import MockEmbeddingProvider
from vertical_brain.storage.sqlite_store import SQLiteStore
from vertical_brain.storage.json_store import JsonStore

MIN_COMPACTION_PATH_PARTS = 5


def build_optimizer(store: JsonStore) -> SimpleOptimizer:
    return SimpleOptimizer(store, min_compaction_path_parts=MIN_COMPACTION_PATH_PARTS)


def test_optimizer_marks_exact_duplicate_chunks_stale(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Delta fact"))
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Delta fact"))
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Streaming fact"))

    report = build_optimizer(store).optimize_branch("WORK/DataArt/Databricks")

    chunks = store.get_chunks_by_path("WORK/DataArt/Databricks")
    assert [chunk.status for chunk in chunks].count("active") == 2
    assert [chunk.status for chunk in chunks].count("stale") == 1
    assert "1 exact duplicates marked stale" in report


def test_optimizer_compacts_related_active_chunks_and_preserves_lineage(tmp_path):
    store = JsonStore(tmp_path)
    first = store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/Databricks/Certification/StructuredStreaming/SchemaEvolution",
            content="Delta streaming sink schema evolution uses mergeSchema=true.",
            content_type="fact",
        )
    )
    second = store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/Databricks/Certification/StructuredStreaming/SchemaEvolution",
            content="Auto Loader schema evolution is separate from Delta sink schema evolution.",
            content_type="correction",
        )
    )
    store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/Databricks/Certification/AutoLoader",
            content="Auto Loader supports incremental file ingestion.",
            content_type="fact",
        )
    )

    report = build_optimizer(store).optimize_branch(
        "WORK/DataArt/Databricks/Certification/StructuredStreaming"
    )

    chunks = store.get_chunks_by_path(
        "WORK/DataArt/Databricks/Certification/StructuredStreaming/SchemaEvolution"
    )
    compacted = [chunk for chunk in chunks if chunk.source == "optimizer:namespace_compaction"]
    originals = [chunk for chunk in chunks if chunk.id in {first.id, second.id}]
    assert len(compacted) == 1
    assert compacted[0].layer == "silver"
    assert compacted[0].status == "active"
    assert compacted[0].lineage == [first.id, second.id]
    assert (
        "Compacted Silver summary for namespace: "
        "WORK/DataArt/Databricks/Certification/StructuredStreaming/SchemaEvolution."
        in compacted[0].content
    )
    assert {chunk.status for chunk in originals} == {"superseded"}
    assert "1 namespace compactions created" in report

    unrelated = store.get_chunks_by_path("WORK/DataArt/Databricks/Certification/AutoLoader")
    assert len(unrelated) == 1
    assert unrelated[0].status == "active"


def test_optimizer_namespace_compaction_is_idempotent(tmp_path):
    store = JsonStore(tmp_path)
    path = "WORK/DataArt/Databricks/Certification/StructuredStreaming/SchemaEvolution"
    store.save_chunk(Chunk(node_path=path, content="Schema evolution uses mergeSchema."))
    store.save_chunk(Chunk(node_path=path, content="Schema evolution applies to Delta writes."))

    optimizer = build_optimizer(store)
    optimizer.optimize_branch(path)
    optimizer.optimize_branch(path)

    chunks = store.get_chunks_by_path(path)
    assert len([chunk for chunk in chunks if chunk.source == "optimizer:namespace_compaction"]) == 1


def test_optimizer_does_not_compact_across_namespaces(tmp_path):
    store = JsonStore(tmp_path)
    branch = "WORK/DataArt/Databricks/Misc"
    store.save_chunk(
        Chunk(
            node_path=f"{branch}/Credentials",
            content="Credentials should be stored in the password manager.",
        )
    )
    store.save_chunk(
        Chunk(
            node_path=f"{branch}/Meetings",
            content="Meeting notes should be reviewed every Friday.",
        )
    )

    report = build_optimizer(store).optimize_branch(branch)

    chunks = store.get_chunks_by_path(branch, include_children=True)
    assert len([chunk for chunk in chunks if chunk.source == "optimizer:namespace_compaction"]) == 0
    assert {chunk.status for chunk in chunks} == {"active"}
    assert "no optimization changes required" in report


# ── no-op behavior ────────────────────────────────────────────────────────────

def test_optimizer_no_op_does_not_call_apply_batch(tmp_path):
    """Empty branch produces no operations — apply_batch must not be called."""
    store = JsonStore(tmp_path)
    # Single chunk at each node: nothing to deduplicate or compact.
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks/Misc/A", content="fact A"))

    with patch(
        "vertical_brain.core.optimizer.StorageOperationExecutor"
    ) as mock_executor_cls:
        build_optimizer(store).optimize_branch("WORK/DataArt/Databricks/Misc")

    mock_executor_cls.return_value.apply_batch.assert_not_called()
    mock_executor_cls.return_value.apply.assert_not_called()


def test_optimizer_no_op_report_says_no_changes_required(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks/Misc/A", content="only fact"))

    report = build_optimizer(store).optimize_branch("WORK/DataArt/Databricks/Misc")

    assert "no optimization changes required" in report


# ── I/O contract guard tests ──────────────────────────────────────────────────

def test_optimizer_reads_store_exactly_once(tmp_path):
    """get_chunks_by_path must be called exactly once per optimize_branch call."""
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks/Misc/A", content="fact one"))
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks/Misc/A", content="fact two"))

    original_get = store.get_chunks_by_path
    call_count = []

    def counting_get(path, **kwargs):
        call_count.append(path)
        return original_get(path, **kwargs)

    store.get_chunks_by_path = counting_get  # type: ignore[method-assign]
    build_optimizer(store).optimize_branch("WORK/DataArt/Databricks/Misc")

    assert len(call_count) == 1


def test_optimizer_calls_apply_batch_exactly_once_when_operations_exist(tmp_path):
    """All planned operations must be applied in a single apply_batch call."""
    store = JsonStore(tmp_path)
    path = "WORK/DataArt/Databricks/Misc/A"
    store.save_chunk(Chunk(node_path=path, content="duplicate fact"))
    store.save_chunk(Chunk(node_path=path, content="duplicate fact"))

    with patch(
        "vertical_brain.core.optimizer.StorageOperationExecutor"
    ) as mock_executor_cls:
        mock_executor_cls.return_value.apply_batch.return_value = MagicMock(results=[])
        build_optimizer(store).optimize_branch("WORK/DataArt/Databricks/Misc")

    mock_executor_cls.return_value.apply_batch.assert_called_once()
    mock_executor_cls.return_value.apply.assert_not_called()


def test_optimizer_never_calls_individual_apply(tmp_path):
    """apply() must never be called — only apply_batch()."""
    store = JsonStore(tmp_path)
    path = "WORK/DataArt/Databricks/Misc/A"
    # Multiple chunks so compaction is also triggered.
    store.save_chunk(Chunk(node_path=path, content="fact alpha"))
    store.save_chunk(Chunk(node_path=path, content="fact beta"))
    store.save_chunk(Chunk(node_path=path, content="fact alpha"))  # duplicate

    with patch(
        "vertical_brain.core.optimizer.StorageOperationExecutor"
    ) as mock_executor_cls:
        mock_executor_cls.return_value.apply_batch.return_value = MagicMock(results=[])
        build_optimizer(store).optimize_branch("WORK/DataArt/Databricks/Misc")

    mock_executor_cls.return_value.apply.assert_not_called()


# ── content_hash dedup ────────────────────────────────────────────────────────

def test_optimizer_deduplicates_by_content_hash_when_present(tmp_path):
    """Dedup key uses content_hash; chunks with the same hash are treated as duplicates."""
    store = SQLiteStore(tmp_path)
    # Save identical content — SQLiteStore populates content_hash automatically.
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="same content"))
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="same content"))

    build_optimizer(store).optimize_branch("WORK/DataArt/Databricks")

    chunks = store.get_chunks_by_path("WORK/DataArt/Databricks")
    active = [c for c in chunks if c.status == "active"]
    stale = [c for c in chunks if c.status == "stale"]
    assert len(active) == 1
    assert len(stale) == 1


def test_optimizer_plan_dedup_key_falls_back_to_content_when_hash_empty(tmp_path):
    """plan_branch dedup must still work when content_hash is absent (e.g. JsonStore)."""
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="identical content"))
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="identical content"))

    plan = SimpleOptimizer(store, min_compaction_path_parts=3).plan_branch("WORK/DataArt")

    stale_ops = [op for op in plan.operations if op.operation == "mark_stale"]
    assert len(stale_ops) == 1


# ── fact decay ────────────────────────────────────────────────────────────────

def test_optimizer_decay_marks_old_unlinked_chunks_stale(tmp_path):
    """Chunks older than decay_days with effective confidence below threshold become stale."""
    from datetime import datetime, timedelta, timezone
    store = JsonStore(tmp_path)
    path = "WORK/DataArt/Databricks/Misc/A"
    old_time = (datetime.now(timezone.utc) - timedelta(days=91)).isoformat()
    chunk = Chunk(node_path=path, content="old unlinked fact")
    chunk.created_at = old_time
    store.save_chunk(chunk)

    optimizer = SimpleOptimizer(
        store,
        min_compaction_path_parts=MIN_COMPACTION_PATH_PARTS,
        decay_rate=0.5,
        decay_days=30,
        stale_threshold=0.2,
    )
    optimizer.optimize_branch("WORK/DataArt/Databricks/Misc")

    chunks = store.get_chunks_by_path(path)
    assert all(c.status == "stale" for c in chunks)


def test_optimizer_decay_does_not_affect_gold_chunks(tmp_path):
    """Gold chunks must never be decayed regardless of age."""
    from datetime import datetime, timedelta, timezone
    store = JsonStore(tmp_path)
    path = "WORK/DataArt/Databricks/Misc/A"
    old_time = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    chunk = Chunk(node_path=path, content="old gold summary", layer="gold")
    chunk.created_at = old_time
    store.save_chunk(chunk)

    optimizer = SimpleOptimizer(
        store,
        min_compaction_path_parts=MIN_COMPACTION_PATH_PARTS,
        decay_rate=0.1,
        decay_days=30,
        stale_threshold=0.5,
    )
    optimizer.optimize_branch("WORK/DataArt/Databricks/Misc")

    chunks = store.get_chunks_by_path(path)
    assert all(c.status == "active" for c in chunks)


def test_optimizer_decay_skips_linked_namespaces(tmp_path):
    """Chunks whose namespace has any link must not be decayed."""
    from datetime import datetime, timedelta, timezone
    from vertical_brain.core.models import Link
    store = JsonStore(tmp_path)
    path = "WORK/DataArt/Databricks/Misc/A"
    old_time = (datetime.now(timezone.utc) - timedelta(days=91)).isoformat()
    chunk = Chunk(node_path=path, content="linked but old fact")
    chunk.created_at = old_time
    store.save_chunk(chunk)
    store.save_link(Link(source_path=path, target_path="WORK/Other", link_type="peer", reason="related"))

    optimizer = SimpleOptimizer(
        store,
        min_compaction_path_parts=MIN_COMPACTION_PATH_PARTS,
        decay_rate=0.5,
        decay_days=30,
        stale_threshold=0.2,
    )
    optimizer.optimize_branch("WORK/DataArt/Databricks/Misc")

    chunks = store.get_chunks_by_path(path)
    assert all(c.status == "active" for c in chunks)


def test_optimizer_decay_default_rate_1_never_decays(tmp_path):
    """With default decay_rate=1.0 no chunks are ever decayed."""
    from datetime import datetime, timedelta, timezone
    store = JsonStore(tmp_path)
    path = "WORK/DataArt/Databricks/Misc/A"
    old_time = (datetime.now(timezone.utc) - timedelta(days=3650)).isoformat()
    chunk = Chunk(node_path=path, content="ancient fact", confidence=0.01)
    chunk.created_at = old_time
    store.save_chunk(chunk)

    build_optimizer(store).optimize_branch("WORK/DataArt/Databricks/Misc")

    chunks = store.get_chunks_by_path(path)
    assert all(c.status == "active" for c in chunks)


# ── optimize_all ──────────────────────────────────────────────────────────────

def test_optimize_all_deduplicates_across_namespaces(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="PROJECTS/alpha", content="duplicate fact"))
    store.save_chunk(Chunk(node_path="PROJECTS/alpha", content="duplicate fact"))
    store.save_chunk(Chunk(node_path="PROJECTS/beta", content="unique fact"))

    report = build_optimizer(store).optimize_all()

    chunks = store.get_chunks_by_path("PROJECTS/alpha")
    assert sum(1 for c in chunks if c.status == "active") == 1
    assert "1 exact duplicates marked stale" in report


def test_optimize_all_includes_link_discovery(tmp_path):
    store = JsonStore(tmp_path)
    text = "machine learning model training pipeline gradient descent"
    store.save_chunk(Chunk(node_path="PROJECTS/alpha", content=text, layer="silver"))
    store.save_chunk(Chunk(node_path="PROJECTS/beta", content=text, layer="silver"))

    provider = MockEmbeddingProvider()
    optimizer = SimpleOptimizer(
        store,
        min_compaction_path_parts=MIN_COMPACTION_PATH_PARTS,
        embedding_provider=provider,
        link_similarity_threshold=0.5,
    )
    report = optimizer.optimize_all()

    assert store.list_links()
    assert "link(s) created" in report


# ── discover_links ─────────────────────────────────────────────────────────────

def test_discover_links_skips_without_provider(tmp_path):
    store = JsonStore(tmp_path)
    result = build_optimizer(store).discover_links()
    assert "skipped" in result.lower()


def test_discover_links_creates_link_for_similar_silver_chunks(tmp_path):
    store = JsonStore(tmp_path)
    # Two Silver chunks in different namespaces with very similar content
    repeated_text = "machine learning model training pipeline gradient descent"
    store.save_chunk(Chunk(node_path="PROJECTS/alpha", content=repeated_text, layer="silver"))
    store.save_chunk(Chunk(node_path="PROJECTS/beta", content=repeated_text, layer="silver"))

    provider = MockEmbeddingProvider()
    optimizer = SimpleOptimizer(
        store,
        min_compaction_path_parts=MIN_COMPACTION_PATH_PARTS,
        embedding_provider=provider,
        link_similarity_threshold=0.5,
    )
    result = optimizer.discover_links()

    links = store.list_links()
    assert len(links) == 1
    assert {links[0].source_path, links[0].target_path} == {"PROJECTS/alpha", "PROJECTS/beta"}
    assert "1 link(s) created" in result


def test_discover_links_skips_existing_link(tmp_path):
    from vertical_brain.core.models import Link
    store = JsonStore(tmp_path)
    text = "machine learning model training pipeline gradient descent"
    store.save_chunk(Chunk(node_path="PROJECTS/alpha", content=text, layer="silver"))
    store.save_chunk(Chunk(node_path="PROJECTS/beta", content=text, layer="silver"))
    store.save_link(Link(source_path="PROJECTS/alpha", target_path="PROJECTS/beta", link_type="related", reason="manual"))

    provider = MockEmbeddingProvider()
    optimizer = SimpleOptimizer(
        store,
        min_compaction_path_parts=MIN_COMPACTION_PATH_PARTS,
        embedding_provider=provider,
        link_similarity_threshold=0.5,
    )
    result = optimizer.discover_links()

    assert len(store.list_links()) == 1  # no new link created
    assert "no cross-namespace" in result.lower()


def test_discover_links_ignores_bronze_and_gold_chunks(tmp_path):
    store = JsonStore(tmp_path)
    text = "machine learning model training pipeline gradient descent"
    store.save_chunk(Chunk(node_path="PROJECTS/alpha", content=text, layer="bronze"))
    store.save_chunk(Chunk(node_path="PROJECTS/beta", content=text, layer="gold"))

    provider = MockEmbeddingProvider()
    optimizer = SimpleOptimizer(
        store,
        min_compaction_path_parts=MIN_COMPACTION_PATH_PARTS,
        embedding_provider=provider,
        link_similarity_threshold=0.5,
    )
    result = optimizer.discover_links()

    assert store.list_links() == []
    assert "fewer than 2" in result or "no cross-namespace" in result.lower()


def test_discover_links_no_link_for_dissimilar_chunks(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="PROJECTS/alpha", content="quantum physics particle accelerator", layer="silver"))
    store.save_chunk(Chunk(node_path="PROJECTS/beta", content="cookie recipe butter flour sugar bake", layer="silver"))

    provider = MockEmbeddingProvider()
    optimizer = SimpleOptimizer(
        store,
        min_compaction_path_parts=MIN_COMPACTION_PATH_PARTS,
        embedding_provider=provider,
        link_similarity_threshold=0.95,
    )
    result = optimizer.discover_links()

    assert store.list_links() == []
    assert "no cross-namespace" in result.lower()
