"""trigger_reindexing tests — cache warm-up and provider migration."""
from __future__ import annotations

from vertical_brain.core.embedding_search import EmbeddingSearch, IncompatibleEmbeddingModelError
from vertical_brain.core.models import Chunk
from vertical_brain.llm.embedding import MockEmbeddingProvider
from vertical_brain.storage.json_store import JsonStore


def test_trigger_reindexing_populates_cache(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="fact one"))
    store.save_chunk(Chunk(node_path="WORK/B", content="fact two"))
    es = EmbeddingSearch(store, MockEmbeddingProvider("m1"), validate_schema=False)
    assert len(es._cache) == 0

    count = es.trigger_reindexing()

    assert count == 2
    assert len(es._cache) == 2


def test_trigger_reindexing_skips_stale_chunks(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="active fact"))
    store.save_chunk(Chunk(node_path="WORK/A", content="stale fact", status="stale"))
    es = EmbeddingSearch(store, MockEmbeddingProvider("m1"), validate_schema=False)

    count = es.trigger_reindexing()

    assert count == 1


def test_trigger_reindexing_scopes_to_node_path(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="in scope"))
    store.save_chunk(Chunk(node_path="OTHER/B", content="out of scope"))
    es = EmbeddingSearch(store, MockEmbeddingProvider("m1"), validate_schema=False)

    count = es.trigger_reindexing(node_path="WORK")

    assert count == 1
    assert "in scope" in es._cache
    assert "out of scope" not in es._cache


def test_trigger_reindexing_with_new_provider_updates_schema(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="fact"))
    es = EmbeddingSearch(store, MockEmbeddingProvider("model-v1"))

    new_provider = MockEmbeddingProvider("model-v2")
    es.trigger_reindexing(new_provider)

    schema = store.get_embedding_schema()
    assert schema["model_name"] == "model-v2"


def test_trigger_reindexing_with_new_provider_clears_old_cache(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="fact"))
    provider_v1 = MockEmbeddingProvider("model-v1")
    es = EmbeddingSearch(store, provider_v1)
    # Pre-populate with old embeddings.
    es._cache["stale_entry"] = [0.1, 0.2]

    provider_v2 = MockEmbeddingProvider("model-v2")
    es.trigger_reindexing(provider_v2)

    assert "stale_entry" not in es._cache


def test_trigger_reindexing_enables_search_with_new_provider(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="Delta Lake streaming ingestion"))
    es = EmbeddingSearch(store, MockEmbeddingProvider("model-v1"))

    es.trigger_reindexing(MockEmbeddingProvider("model-v2"))
    results = es.search("Delta Lake streaming")

    assert len(results) >= 1


def test_trigger_reindexing_without_new_provider_keeps_provider(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="fact"))
    provider = MockEmbeddingProvider("model-v1")
    es = EmbeddingSearch(store, provider)

    es.trigger_reindexing()

    assert es._provider is provider
