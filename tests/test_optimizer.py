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
    assert "Gold summary placeholder for WORK/DataArt/Databricks." in node.gold_summary
    assert "Active chunks: 2." in node.gold_summary

    gold_file = store.gold_summary_path("WORK/DataArt/Databricks")
    assert gold_file.exists()
    assert gold_file.read_text(encoding="utf-8").startswith("# WORK/DataArt/Databricks")
