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


def test_incompatible_model_raises_in_sqlite(tmp_path):
    store = SQLiteStore(tmp_path)
    EmbeddingSearch(store, MockEmbeddingProvider("model-A"))

    with pytest.raises(IncompatibleEmbeddingModelError, match="model-B"):
        EmbeddingSearch(store, MockEmbeddingProvider("model-B"))
