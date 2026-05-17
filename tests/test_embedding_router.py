import pytest

from vertical_brain.core.embedding_router import EmbeddingRouter
from vertical_brain.core.models import Chunk
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.core.models import StorageOperation
from vertical_brain.llm.embedding import MockEmbeddingProvider
from vertical_brain.storage.json_store import JsonStore


class KeywordEmbeddingProvider:
    def __init__(self) -> None:
        self.seen = []

    def embed(self, text: str):
        self.seen.append(text)
        lower = text.lower()
        return [
            1.0 if "z-ordering" in lower else 0.0,
            1.0 if "autoloader" in lower else 0.0,
            1.0 if "mlflow" in lower else 0.0,
        ]


class NamedKeywordEmbeddingProvider(KeywordEmbeddingProvider):
    model_name = "keyword-v1"
    embed_dimension = 3


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
    from vertical_brain.core.models import Chunk
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="silver summary", layer="silver"))
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/DataArt", gold_aspect="Databricks Delta ingestion"))
    executor.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/DataArt", gold_aspect="AutoLoader streaming"))

    candidates = EmbeddingRouter(store, MockEmbeddingProvider()).find_candidates("Delta AutoLoader streaming")

    assert len(candidates) == 1
    assert candidates[0].path == "WORK/DataArt"
    assert "AutoLoader" in candidates[0].gold_summary


def test_embedding_router_scores_each_gold_aspect_independently(tmp_path):
    from vertical_brain.core.gold import GoldAspect, serialize_gold_aspects

    store = JsonStore(tmp_path)
    content = serialize_gold_aspects([
        GoldAspect(text="Delta Z-ordering"),
        GoldAspect(text="AutoLoader schema drift"),
        GoldAspect(text="MLflow tracking"),
    ])
    _seed_gold(store, "WORK/DataArt/Databricks", content)
    provider = KeywordEmbeddingProvider()

    candidates = EmbeddingRouter(store, provider).find_candidates("where to store Z-ordering notes")

    assert candidates[0].path == "WORK/DataArt/Databricks"
    assert candidates[0].score == pytest.approx(1.0)
    assert candidates[0].gold_summary == "Delta Z-ordering"
    assert "Delta Z-ordering" in provider.seen
    assert "Delta Z-ordering | AutoLoader schema drift | MLflow tracking" not in provider.seen


def test_embedding_router_persists_vectors_per_gold_aspect(tmp_path):
    from vertical_brain.core.gold import GoldAspect, gold_aspect_embed_key, serialize_gold_aspects

    store = JsonStore(tmp_path)
    content = serialize_gold_aspects([
        GoldAspect(text="Delta Z-ordering"),
        GoldAspect(text="AutoLoader schema drift"),
    ])
    _seed_gold(store, "WORK/DataArt/Databricks", content)
    provider = NamedKeywordEmbeddingProvider()

    EmbeddingRouter(store, provider).find_candidates("where to store Z-ordering notes")

    z_key = gold_aspect_embed_key("Delta Z-ordering")
    autoloader_key = gold_aspect_embed_key("AutoLoader schema drift")
    assert store.get_vector(z_key, "keyword-v1") == [1.0, 0.0, 0.0]
    assert store.get_vector(autoloader_key, "keyword-v1") == [0.0, 1.0, 0.0]


def test_embedding_router_reuses_persisted_gold_aspect_vectors(tmp_path):
    from vertical_brain.core.gold import GoldAspect, serialize_gold_aspects

    store = JsonStore(tmp_path)
    content = serialize_gold_aspects([
        GoldAspect(text="Delta Z-ordering"),
        GoldAspect(text="AutoLoader schema drift"),
    ])
    _seed_gold(store, "WORK/DataArt/Databricks", content)
    provider = NamedKeywordEmbeddingProvider()
    EmbeddingRouter(store, provider).find_candidates("where to store Z-ordering notes")
    provider.seen = []

    EmbeddingRouter(store, provider).find_candidates("where to store Z-ordering notes")

    assert provider.seen == ["where to store Z-ordering notes"]


def test_embedding_router_recomputes_bad_cached_gold_aspect_vector(tmp_path):
    from vertical_brain.core.gold import GoldAspect, gold_aspect_embed_key, serialize_gold_aspects

    store = JsonStore(tmp_path)
    content = serialize_gold_aspects([GoldAspect(text="Delta Z-ordering")])
    _seed_gold(store, "WORK/DataArt/Databricks", content)
    store.set_vector(gold_aspect_embed_key("Delta Z-ordering"), "keyword-v1", [0.1])
    provider = NamedKeywordEmbeddingProvider()

    EmbeddingRouter(store, provider).find_candidates("where to store Z-ordering notes")

    assert "Delta Z-ordering" in provider.seen
    assert store.get_vector(gold_aspect_embed_key("Delta Z-ordering"), "keyword-v1") == [1.0, 0.0, 0.0]


def test_embedding_router_finds_namespace_with_gold_chunk(tmp_path):
    store = JsonStore(tmp_path)
    _seed_gold(store, "WORK/DataArt/Databricks", "Databricks Delta Lake streaming")

    candidates = EmbeddingRouter(store, MockEmbeddingProvider()).find_candidates("Delta Lake")

    assert any(c.path == "WORK/DataArt/Databricks" for c in candidates)
    match = next(c for c in candidates if c.path == "WORK/DataArt/Databricks")
    assert match.gold_summary != ""


def test_embedding_router_fallback_finds_empty_namespace_by_path_keyword(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/DataArt/Databricks")
    # No Gold chunk — only path match should surface this namespace.

    candidates = EmbeddingRouter(store, MockEmbeddingProvider()).find_candidates("Databricks")

    assert any(c.path == "WORK/DataArt/Databricks" for c in candidates)
    match = next(c for c in candidates if c.path == "WORK/DataArt/Databricks")
    assert match.gold_summary == ""
    assert 0 < match.score <= 0.45


def test_embedding_router_semantic_gold_outranks_path_match(tmp_path):
    store = JsonStore(tmp_path)
    # Gold chunk: semantically very close to query.
    _seed_gold(store, "WORK/DataArt/Databricks", "Databricks Delta Lake autoloader streaming ingestion")
    # Empty namespace whose path contains "Databricks" but has no Gold.
    store.ensure_node("WORK/Other/Databricks")

    candidates = EmbeddingRouter(store, MockEmbeddingProvider()).find_candidates(
        "Databricks Delta streaming"
    )

    paths = [c.path for c in candidates]
    assert "WORK/DataArt/Databricks" in paths
    assert "WORK/Other/Databricks" in paths
    gold_rank = paths.index("WORK/DataArt/Databricks")
    path_rank = paths.index("WORK/Other/Databricks")
    assert gold_rank < path_rank, "semantic Gold score must outrank pure path match"


# ── camelCase / PascalCase path tokenization ─────────────────────────────────

@pytest.mark.parametrize("query", [
    "auto loader",
    "auto-loader",
    "AutoLoader",
])
def test_router_path_fallback_matches_autoloader_namespace(tmp_path, query):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/DataArt/Databricks/AutoLoader")

    candidates = EmbeddingRouter(store, MockEmbeddingProvider()).find_candidates(query)

    assert any(c.path == "WORK/DataArt/Databricks/AutoLoader" for c in candidates)


@pytest.mark.parametrize("query", [
    "structured streaming",
    "structuredStreaming",
    "StructuredStreaming",
])
def test_router_path_fallback_matches_structuredstreaming_namespace(tmp_path, query):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/DataArt/Databricks/StructuredStreaming")

    candidates = EmbeddingRouter(store, MockEmbeddingProvider()).find_candidates(query)

    assert any(c.path == "WORK/DataArt/Databricks/StructuredStreaming" for c in candidates)


# ── Gold embed text: clean aspect text, not raw JSON ─────────────────────────

def test_router_embeds_clean_gold_text_not_raw_json(tmp_path):
    """Router gold_summary must contain clean aspect text, not JSON structure or UUIDs."""
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="silver summary", layer="silver"))
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/DataArt", gold_aspect="Delta Lake autoloader schema drift detection"))

    candidates = EmbeddingRouter(store, MockEmbeddingProvider()).find_candidates("Delta Lake autoloader")

    assert len(candidates) == 1
    summary = candidates[0].gold_summary
    # Must contain the actual aspect text
    assert "Delta Lake" in summary
    # Must NOT contain JSON noise
    assert '"id"' not in summary
    assert '"updated_at"' not in summary
    assert '"aspects"' not in summary


def test_router_gold_embed_text_excludes_uuid_and_timestamps(tmp_path):
    """Gold chunk stored as v2 JSON must expose clean aspect text."""
    from vertical_brain.core.gold import serialize_gold_aspects, GoldAspect
    from vertical_brain.core.models import Chunk

    store = JsonStore(tmp_path)
    # Build a v2 Gold chunk directly with known UUIDs and timestamps
    aspects = [
        GoldAspect(id="aaaaaaaa-0000-0000-0000-000000000001", text="Fact one about streaming"),
        GoldAspect(id="bbbbbbbb-0000-0000-0000-000000000002", text="Fact two about Delta Lake"),
    ]
    gold_content = serialize_gold_aspects(aspects)
    store.ensure_node("WORK/Topic")
    store.save_chunk(Chunk(node_path="WORK/Topic", content=gold_content, layer="gold"))

    candidates = EmbeddingRouter(store, MockEmbeddingProvider()).find_candidates("streaming Delta Lake")

    assert len(candidates) == 1
    summary = candidates[0].gold_summary
    assert summary in {"Fact one about streaming", "Fact two about Delta Lake"}
    assert "aaaaaaaa" not in summary
    assert "bbbbbbbb" not in summary
