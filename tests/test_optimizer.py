from vertical_brain.core.models import Chunk
from vertical_brain.core.optimizer import SimpleOptimizer
from vertical_brain.storage.json_store import JsonStore


def test_optimizer_marks_exact_duplicate_chunks_stale_and_updates_gold_summary(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Delta fact"))
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Delta fact"))
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Streaming fact"))

    report = SimpleOptimizer(store).optimize_branch("WORK/DataArt/Databricks")

    chunks = store.get_chunks_by_path("WORK/DataArt/Databricks")
    assert [chunk.status for chunk in chunks].count("active") == 2
    assert [chunk.status for chunk in chunks].count("stale") == 1
    assert "1 exact duplicates marked stale" in report
    assert "Gold summary file:" in report

    node = store.get_node("WORK/DataArt/Databricks")
    assert node is not None
    assert "Gold summary for WORK/DataArt/Databricks." in node.gold_summary
    assert "Active chunks: 2." in node.gold_summary

    gold_file = store.gold_summary_path("WORK/DataArt/Databricks")
    assert gold_file.exists()
    assert gold_file.read_text(encoding="utf-8").startswith("# WORK/DataArt/Databricks")


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

    report = SimpleOptimizer(store).optimize_branch(
        "WORK/DataArt/Databricks/Certification/StructuredStreaming"
    )

    chunks = store.get_chunks_by_path(
        "WORK/DataArt/Databricks/Certification/StructuredStreaming/SchemaEvolution"
    )
    compacted = [chunk for chunk in chunks if chunk.source == "optimizer:semantic_compaction"]
    originals = [chunk for chunk in chunks if chunk.id in {first.id, second.id}]
    assert len(compacted) == 1
    assert compacted[0].layer == "silver"
    assert compacted[0].status == "active"
    assert compacted[0].lineage == [first.id, second.id]
    assert {chunk.status for chunk in originals} == {"superseded"}
    assert "1 semantic compactions created" in report

    unrelated = store.get_chunks_by_path("WORK/DataArt/Databricks/Certification/AutoLoader")
    assert len(unrelated) == 1
    assert unrelated[0].status == "active"


def test_optimizer_semantic_compaction_is_idempotent(tmp_path):
    store = JsonStore(tmp_path)
    path = "WORK/DataArt/Databricks/Certification/StructuredStreaming/SchemaEvolution"
    store.save_chunk(Chunk(node_path=path, content="Schema evolution uses mergeSchema."))
    store.save_chunk(Chunk(node_path=path, content="Schema evolution applies to Delta writes."))

    optimizer = SimpleOptimizer(store)
    optimizer.optimize_branch(path)
    optimizer.optimize_branch(path)

    chunks = store.get_chunks_by_path(path)
    assert len([chunk for chunk in chunks if chunk.source == "optimizer:semantic_compaction"]) == 1
