"""Embedding schema versioning tests."""
from __future__ import annotations

import pytest

from vertical_brain.core.embedding_search import EmbeddingSearch, IncompatibleEmbeddingModelError
from vertical_brain.llm.embedding import MockEmbeddingProvider
from vertical_brain.storage.json_store import JsonStore
from vertical_brain.storage.sqlite_store import SQLiteStore


def test_first_search_registers_schema(tmp_path):
    store = JsonStore(tmp_path)
    provider = MockEmbeddingProvider("test-model-v1")
    EmbeddingSearch(store, provider)

    schema = store.get_embedding_schema()
    assert schema is not None
    assert schema["model_name"] == "test-model-v1"


def test_compatible_model_does_not_raise(tmp_path):
    store = JsonStore(tmp_path)
    provider = MockEmbeddingProvider("test-model-v1")
    EmbeddingSearch(store, provider)
    # Second instance with same model must succeed.
    EmbeddingSearch(store, provider)


def test_incompatible_model_raises(tmp_path):
    store = JsonStore(tmp_path)
    EmbeddingSearch(store, MockEmbeddingProvider("model-A"))

    with pytest.raises(IncompatibleEmbeddingModelError, match="model-B"):
        EmbeddingSearch(store, MockEmbeddingProvider("model-B"))


def test_validate_schema_false_skips_check(tmp_path):
    store = JsonStore(tmp_path)
    EmbeddingSearch(store, MockEmbeddingProvider("model-A"))
    # Must not raise even with a different model when validation is disabled.
    EmbeddingSearch(store, MockEmbeddingProvider("model-B"), validate_schema=False)


def test_schema_persisted_in_sqlite(tmp_path):
    store = SQLiteStore(tmp_path)
    provider = MockEmbeddingProvider("sqlite-model-v1")
    EmbeddingSearch(store, provider)

    schema = store.get_embedding_schema()
    assert schema is not None
    assert schema["model_name"] == "sqlite-model-v1"


def test_old_sqlite_embedding_schema_table_migrates(tmp_path):
    import sqlite3

    conn = sqlite3.connect(tmp_path / "vertical_brain.sqlite")
    conn.executescript(
        """
        CREATE TABLE embedding_schema (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            model_name TEXT NOT NULL,
            vector_dimension INTEGER NOT NULL
        );
        """
    )
    conn.execute("INSERT INTO embedding_schema VALUES (1, ?, ?)", ("old-model", 3))
    conn.commit()
    conn.close()

    store = SQLiteStore(tmp_path)
    store.set_embedding_schema("new-model", 64)

    schema = store.get_embedding_schema()
    assert schema is not None
    assert schema["model_name"] == "new-model"
    assert schema["vector_dimension"] == 64
    assert "updated_at" in schema


def test_incompatible_model_raises_in_sqlite(tmp_path):
    store = SQLiteStore(tmp_path)
    EmbeddingSearch(store, MockEmbeddingProvider("model-A"))

    with pytest.raises(IncompatibleEmbeddingModelError, match="model-B"):
        EmbeddingSearch(store, MockEmbeddingProvider("model-B"))
