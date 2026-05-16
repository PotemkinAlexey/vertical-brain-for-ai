from vertical_brain.core.embedding_search import EmbeddingSearch
from vertical_brain.core.models import Chunk
from vertical_brain.llm.embedding import MockEmbeddingProvider
from vertical_brain.storage.json_store import JsonStore
from vertical_brain.storage.sqlite_store import SQLiteStore


def test_semantic_search_returns_most_similar_chunk(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Delta Lake streaming autoloader ingestion"))
    store.save_chunk(Chunk(node_path="WORK/DataArt/Python", content="Python pandas polars data frames"))

    results = EmbeddingSearch(store, MockEmbeddingProvider()).search("Delta Lake streaming")

    assert len(results) >= 1
    assert results[0].path == "WORK/DataArt/Databricks"
    assert results[0].score > 0
    assert results[0].source == "chunk:semantic"


def test_semantic_search_excludes_stale_by_default(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="Delta Lake streaming", status="stale"))

    results = EmbeddingSearch(store, MockEmbeddingProvider()).search("Delta Lake streaming")

    assert results == []


def test_semantic_search_includes_stale_when_requested(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="Delta Lake streaming", status="stale"))

    results = EmbeddingSearch(store, MockEmbeddingProvider()).search(
        "Delta Lake streaming", include_stale=True
    )

    assert len(results) == 1
    assert results[0].status == "stale"


def test_semantic_search_respects_root_path_scope(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Delta Lake streaming"))
    store.save_chunk(Chunk(node_path="PERSONAL/Blog", content="Delta Lake streaming blog post"))

    results = EmbeddingSearch(store, MockEmbeddingProvider()).search(
        "Delta Lake streaming", root_path="WORK"
    )

    paths = {r.path for r in results}
    assert "PERSONAL/Blog" not in paths
    assert "WORK/DataArt/Databricks" in paths


def test_semantic_search_threshold_filters_low_scores(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="Delta Lake streaming"))

    # cosine similarity is at most 1.0, so threshold above 1.0 must always filter everything
    results = EmbeddingSearch(store, MockEmbeddingProvider()).search(
        "Delta Lake streaming", threshold=1.01
    )

    assert results == []


def test_semantic_search_limit_caps_results(tmp_path):
    store = JsonStore(tmp_path)
    for i in range(8):
        store.save_chunk(Chunk(node_path=f"WORK/NS{i}", content=f"topic area subject domain content {i}"))

    results = EmbeddingSearch(store, MockEmbeddingProvider()).search("topic content", limit=3)

    assert len(results) <= 3


def test_semantic_search_ranks_by_descending_score(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="neural networks deep learning python tensorflow"))
    store.save_chunk(Chunk(node_path="WORK/B", content="financial market trading orders execution"))
    store.save_chunk(Chunk(node_path="WORK/C", content="python machine learning scikit models"))

    results = EmbeddingSearch(store, MockEmbeddingProvider()).search("python deep learning neural")

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_semantic_search_returns_correct_metadata(tmp_path):
    store = JsonStore(tmp_path)
    chunk = store.save_chunk(Chunk(
        node_path="WORK/DataArt",
        content="Delta Lake fact",
        layer="silver",
        content_type="fact",
    ))

    results = EmbeddingSearch(store, MockEmbeddingProvider()).search("Delta Lake")

    assert len(results) == 1
    assert results[0].chunk_id == chunk.id
    assert results[0].layer == "silver"
    assert results[0].content_type == "fact"


def test_semantic_search_truncates_long_snippet(tmp_path):
    store = JsonStore(tmp_path)
    long_content = "Delta Lake streaming ingestion fact. " * 20
    store.save_chunk(Chunk(node_path="WORK/DataArt", content=long_content))

    results = EmbeddingSearch(store, MockEmbeddingProvider()).search("Delta Lake")

    assert len(results[0].snippet) <= 163  # 160 + "..."
    assert results[0].snippet.endswith("...")


# ── vector sovereignty ────────────────────────────────────────────────────────

def test_embedding_search_root_path_excludes_other_verticals(tmp_path):
    """A vector query scoped to one vertical must not return chunks from another."""
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Delta Lake streaming ingestion"))
    store.save_chunk(Chunk(node_path="TRADING/Bots", content="Delta Lake strategy execution"))

    results = EmbeddingSearch(store, MockEmbeddingProvider()).search(
        "Delta Lake streaming",
        root_path="WORK/DataArt",
    )

    paths = [r.path for r in results]
    assert "WORK/DataArt/Databricks" in paths
    assert "TRADING/Bots" not in paths


def test_embedding_search_without_root_path_spans_all_verticals(tmp_path):
    """Without a root_path restriction, search returns results from any namespace."""
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Delta Lake streaming ingestion"))
    store.save_chunk(Chunk(node_path="TRADING/Bots", content="Delta Lake strategy execution"))

    results = EmbeddingSearch(store, MockEmbeddingProvider()).search("Delta Lake")

    paths = [r.path for r in results]
    assert "WORK/DataArt/Databricks" in paths
    assert "TRADING/Bots" in paths


def test_semantic_search_scoped_uses_backend_path_filter(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Delta Lake streaming ingestion"))
    store.save_chunk(Chunk(node_path="TRADING/Bots", content="Delta Lake strategy execution"))

    def fail_list_chunks():
        raise AssertionError("scoped semantic search must not scan every chunk")

    store.list_chunks = fail_list_chunks  # type: ignore[method-assign]

    results = EmbeddingSearch(store, MockEmbeddingProvider()).search(
        "Delta Lake streaming",
        root_path="WORK",
    )

    assert [r.path for r in results] == ["WORK/DataArt/Databricks"]


def test_trigger_reindexing_scoped_uses_backend_path_filter(tmp_path):
    store = SQLiteStore(tmp_path)
    in_scope = store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Delta Lake streaming ingestion"))
    out_scope = store.save_chunk(Chunk(node_path="TRADING/Bots", content="Delta Lake strategy execution"))

    def fail_list_chunks():
        raise AssertionError("scoped reindexing must not scan every chunk")

    store.list_chunks = fail_list_chunks  # type: ignore[method-assign]

    indexed = EmbeddingSearch(store, MockEmbeddingProvider()).trigger_reindexing(node_path="WORK")

    assert indexed == 1
    assert store.get_vector(in_scope.content_hash, "mock-v1") is not None
    assert store.get_vector(out_scope.content_hash, "mock-v1") is None
