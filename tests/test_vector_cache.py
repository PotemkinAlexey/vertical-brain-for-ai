"""Persistent vector cache tests — storage backends and EmbeddingSearch integration."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from vertical_brain.core.embedding_search import EmbeddingSearch
from vertical_brain.core.models import Chunk
from vertical_brain.llm.embedding import MockEmbeddingProvider
from vertical_brain.storage.json_store import JsonStore
from vertical_brain.storage.sqlite_store import SQLiteStore


# ── JsonStore vector cache ────────────────────────────────────────────────────

def test_json_store_set_and_get_vector(tmp_path):
    store = JsonStore(tmp_path)
    store.set_vector("abc123", "mock-v1", [0.1, 0.2, 0.3])
    assert store.get_vector("abc123", "mock-v1") == [0.1, 0.2, 0.3]


def test_json_store_get_vector_missing_returns_none(tmp_path):
    store = JsonStore(tmp_path)
    assert store.get_vector("nonexistent", "mock-v1") is None


def test_json_store_get_vector_wrong_model_returns_none(tmp_path):
    store = JsonStore(tmp_path)
    store.set_vector("abc123", "mock-v1", [0.1, 0.2])
    assert store.get_vector("abc123", "other-model") is None


def test_json_store_set_vector_overwrites_existing(tmp_path):
    store = JsonStore(tmp_path)
    store.set_vector("abc123", "mock-v1", [0.1, 0.2])
    store.set_vector("abc123", "mock-v1", [0.9, 0.8])
    assert store.get_vector("abc123", "mock-v1") == [0.9, 0.8]


def test_json_store_delete_vectors_for_model(tmp_path):
    store = JsonStore(tmp_path)
    store.set_vector("h1", "mock-v1", [0.1])
    store.set_vector("h2", "mock-v1", [0.2])
    store.set_vector("h1", "other-model", [0.5])

    deleted = store.delete_vectors_for_model("mock-v1")
    assert deleted == 2
    assert store.get_vector("h1", "mock-v1") is None
    assert store.get_vector("h2", "mock-v1") is None
    assert store.get_vector("h1", "other-model") == [0.5]


def test_json_store_vector_cache_survives_restart(tmp_path):
    store1 = JsonStore(tmp_path)
    store1.set_vector("abc123", "mock-v1", [0.1, 0.2, 0.3])

    store2 = JsonStore(tmp_path)
    assert store2.get_vector("abc123", "mock-v1") == [0.1, 0.2, 0.3]


# ── SQLiteStore vector cache ──────────────────────────────────────────────────

def test_sqlite_store_set_and_get_vector(tmp_path):
    store = SQLiteStore(tmp_path / "brain.sqlite")
    store.set_vector("abc123", "mock-v1", [0.1, 0.2, 0.3])
    assert store.get_vector("abc123", "mock-v1") == [0.1, 0.2, 0.3]


def test_sqlite_store_get_vector_missing_returns_none(tmp_path):
    store = SQLiteStore(tmp_path / "brain.sqlite")
    assert store.get_vector("nonexistent", "mock-v1") is None


def test_sqlite_store_set_vector_upsert(tmp_path):
    store = SQLiteStore(tmp_path / "brain.sqlite")
    store.set_vector("abc123", "mock-v1", [0.1])
    store.set_vector("abc123", "mock-v1", [0.9])
    assert store.get_vector("abc123", "mock-v1") == [0.9]


def test_sqlite_store_delete_vectors_for_model(tmp_path):
    store = SQLiteStore(tmp_path / "brain.sqlite")
    store.set_vector("h1", "mock-v1", [0.1])
    store.set_vector("h2", "mock-v1", [0.2])
    store.set_vector("h1", "other-model", [0.5])

    deleted = store.delete_vectors_for_model("mock-v1")
    assert deleted == 2
    assert store.get_vector("h1", "mock-v1") is None
    assert store.get_vector("h1", "other-model") == [0.5]


def test_sqlite_store_vector_cache_survives_restart(tmp_path):
    db_path = tmp_path / "brain.sqlite"
    store1 = SQLiteStore(db_path)
    store1.set_vector("abc123", "mock-v1", [0.1, 0.2, 0.3])
    del store1

    store2 = SQLiteStore(db_path)
    assert store2.get_vector("abc123", "mock-v1") == [0.1, 0.2, 0.3]


def test_old_sqlite_vector_cache_table_migrates(tmp_path):
    import sqlite3

    conn = sqlite3.connect(tmp_path / "vertical_brain.sqlite")
    conn.executescript(
        """
        CREATE TABLE vector_cache (
            content_hash TEXT NOT NULL,
            model_name TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            PRIMARY KEY (content_hash, model_name)
        );
        """
    )
    conn.execute("INSERT INTO vector_cache VALUES (?, ?, ?)", ("h1", "mock-v1", "[0.1]"))
    conn.commit()
    conn.close()

    store = SQLiteStore(tmp_path)
    store.set_vector("h2", "mock-v1", [0.2])

    assert store.get_vector("h1", "mock-v1") == [0.1]
    assert store.get_vector("h2", "mock-v1") == [0.2]


# ── EmbeddingSearch cache integration ────────────────────────────────────────

def test_embedding_search_stores_vector_on_miss(tmp_path):
    store = JsonStore(tmp_path)
    chunk = store.save_chunk(Chunk(node_path="WORK/A", content="Delta Lake streaming"))
    provider = MockEmbeddingProvider()

    EmbeddingSearch(store, provider).search("Delta Lake")

    cached = store.get_vector(chunk.content_hash, "mock-v1")
    assert cached is not None
    assert len(cached) > 0


def test_embedding_search_reads_from_cache_on_hit(tmp_path):
    store = JsonStore(tmp_path)
    chunk = store.save_chunk(Chunk(node_path="WORK/A", content="Delta Lake streaming"))
    provider = MockEmbeddingProvider()

    # warm the cache
    EmbeddingSearch(store, provider).search("Delta Lake")

    # second search: embed must not be called for chunk content
    with patch.object(provider, "embed", wraps=provider.embed) as mock_embed:
        EmbeddingSearch(store, provider).search("Delta Lake")
        # embed called once for the query, not again for the chunk
        calls = [call.args[0] for call in mock_embed.call_args_list]
        assert chunk.content not in calls


def test_embedding_search_recomputes_wrong_dimension_cached_vector(tmp_path):
    store = JsonStore(tmp_path)
    chunk = store.save_chunk(Chunk(node_path="WORK/A", content="Delta Lake streaming"))
    store.set_vector(chunk.content_hash, "mock-v1", [0.1])
    provider = MockEmbeddingProvider()

    with patch.object(provider, "embed", wraps=provider.embed) as mock_embed:
        EmbeddingSearch(store, provider).search("Delta Lake")

    calls = [call.args[0] for call in mock_embed.call_args_list]
    assert chunk.content in calls
    assert len(store.get_vector(chunk.content_hash, "mock-v1")) == provider.embed_dimension


def test_embedding_search_recomputes_nonnumeric_cached_vector(tmp_path):
    store = JsonStore(tmp_path)
    chunk = store.save_chunk(Chunk(node_path="WORK/A", content="Delta Lake streaming"))
    store.set_vector(chunk.content_hash, "mock-v1", ["bad"])  # type: ignore[list-item]
    provider = MockEmbeddingProvider()

    with patch.object(provider, "embed", wraps=provider.embed) as mock_embed:
        EmbeddingSearch(store, provider).search("Delta Lake")

    calls = [call.args[0] for call in mock_embed.call_args_list]
    assert chunk.content in calls
    assert len(store.get_vector(chunk.content_hash, "mock-v1")) == provider.embed_dimension


def test_embedding_search_rejects_invalid_provider_vector(tmp_path):
    class BadProvider:
        model_name = "bad-provider"
        embed_dimension = 3

        def embed(self, text: str) -> list:
            return [1.0, "bad", 3.0]

    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="Delta Lake streaming"))

    with pytest.raises(ValueError, match="invalid vector"):
        EmbeddingSearch(store, BadProvider()).search("Delta Lake")


def test_trigger_reindexing_warms_persistent_cache(tmp_path):
    store = JsonStore(tmp_path)
    chunk = store.save_chunk(Chunk(node_path="WORK/A", content="Delta Lake streaming"))
    provider = MockEmbeddingProvider()

    es = EmbeddingSearch(store, provider)
    es.trigger_reindexing()

    assert store.get_vector(chunk.content_hash, "mock-v1") is not None


def test_trigger_reindexing_with_new_provider_clears_old_vectors(tmp_path):
    store = JsonStore(tmp_path)
    chunk = store.save_chunk(Chunk(node_path="WORK/A", content="Delta Lake streaming"))

    old_provider = MockEmbeddingProvider("old-model")
    store.set_vector(chunk.content_hash, "old-model", [0.1, 0.2])

    new_provider = MockEmbeddingProvider("new-model")
    es = EmbeddingSearch(store, old_provider, validate_schema=False)
    es.trigger_reindexing(new_provider=new_provider)

    assert store.get_vector(chunk.content_hash, "old-model") is None
    assert store.get_vector(chunk.content_hash, "new-model") is not None


def test_trigger_reindexing_with_new_provider_updates_schema(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="fact"))

    old_provider = MockEmbeddingProvider("old-model")
    store.set_embedding_schema("old-model", 3)

    new_provider = MockEmbeddingProvider("new-model")
    es = EmbeddingSearch(store, old_provider, validate_schema=False)
    es.trigger_reindexing(new_provider=new_provider)

    schema = store.get_embedding_schema()
    assert schema is not None
    assert schema["model_name"] == "new-model"
