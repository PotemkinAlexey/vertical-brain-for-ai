from vertical_brain.core.embedding_router import EmbeddingRouter
from vertical_brain.core.models import Chunk
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.core.models import StorageOperation
from vertical_brain.llm.embedding import MockEmbeddingProvider
from vertical_brain.storage.json_store import JsonStore


def _seed_gold(store, path: str, content: str) -> None:
    store.ensure_node(path)
    store.save_chunk(Chunk(node_path=path, content=content, layer="gold", source="model"))


def test_embedding_router_returns_most_similar_namespace(tmp_path):
    store = JsonStore(tmp_path)
    _seed_gold(store, "WORK/DataArt/Databricks", "Databricks Delta Lake autoloader streaming ingestion")
    _seed_gold(store, "WORK/DataArt/Python", "Python data engineering pandas polars scripts")

    candidates = EmbeddingRouter(store, MockEmbeddingProvider()).find_candidates(
        "Delta Lake streaming job"
    )

    assert len(candidates) >= 1
    assert candidates[0].path == "WORK/DataArt/Databricks"
    assert candidates[0].score > 0


def test_embedding_router_threshold_filters_low_scores(tmp_path):
    store = JsonStore(tmp_path)
    _seed_gold(store, "WORK/DataArt/Databricks", "Databricks Delta Lake streaming ingestion")

    candidates = EmbeddingRouter(store, MockEmbeddingProvider()).find_candidates(
        "Delta Lake streaming job",
        threshold=0.99,
    )

    assert candidates == []


def test_embedding_router_limit_caps_results(tmp_path):
    store = JsonStore(tmp_path)
    for i in range(6):
        _seed_gold(store, f"WORK/NS{i}", f"namespace {i} topic area subject domain")

    candidates = EmbeddingRouter(store, MockEmbeddingProvider()).find_candidates(
        "namespace topic",
        limit=3,
    )

    assert len(candidates) <= 3


def test_embedding_router_returns_empty_when_no_gold_chunks(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="raw bronze fact", layer="bronze"))

    candidates = EmbeddingRouter(store, MockEmbeddingProvider()).find_candidates("anything")

    assert candidates == []


def test_embedding_router_ranks_by_descending_score(tmp_path):
    store = JsonStore(tmp_path)
    _seed_gold(store, "WORK/A", "python machine learning neural networks deep learning")
    _seed_gold(store, "WORK/B", "financial trading market orders execution")
    _seed_gold(store, "WORK/C", "python scripting automation deployment pipelines")

    candidates = EmbeddingRouter(store, MockEmbeddingProvider()).find_candidates("python deep learning")

    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_embedding_router_uses_append_gold_aspect_chunks(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/DataArt", gold_aspect="Databricks Delta ingestion"))
    executor.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/DataArt", gold_aspect="AutoLoader streaming"))

    candidates = EmbeddingRouter(store, MockEmbeddingProvider()).find_candidates("Delta AutoLoader streaming")

    assert len(candidates) == 1
    assert candidates[0].path == "WORK/DataArt"
    assert "AutoLoader" in candidates[0].gold_summary
