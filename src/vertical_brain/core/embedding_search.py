from __future__ import annotations

from typing import TYPE_CHECKING

from vertical_brain.core.models import SearchResult
from vertical_brain.core.search import _path_in_scope
from vertical_brain.llm.embedding import EmbeddingProvider, cosine_similarity

if TYPE_CHECKING:
    from vertical_brain.storage.protocol import StorageProvider


class IncompatibleEmbeddingModelError(Exception):
    """Raised when the embedding provider is incompatible with the indexed schema."""


class EmbeddingSearch:
    """Semantic chunk search using cosine similarity against embedded chunk content."""

    def __init__(
        self,
        store: "StorageProvider",
        provider: EmbeddingProvider,
        *,
        validate_schema: bool = True,
    ) -> None:
        self._store = store
        self._provider = provider
        if validate_schema:
            self._init_embedding_schema()

    def _init_embedding_schema(self) -> None:
        model_name = getattr(self._provider, "model_name", None)
        if model_name is None:
            return
        dim = getattr(self._provider, "embed_dimension", -1)
        get_schema = getattr(self._store, "get_embedding_schema", None)
        set_schema = getattr(self._store, "set_embedding_schema", None)
        if not callable(get_schema) or not callable(set_schema):
            return
        stored = get_schema()
        if stored is None:
            set_schema(model_name, dim)
            return
        if stored["model_name"] != model_name:
            raise IncompatibleEmbeddingModelError(
                f"Provider model '{model_name}' is incompatible with indexed model "
                f"'{stored['model_name']}'. Rebuild the embedding index before searching."
            )
        if dim > 0 and stored.get("vector_dimension", dim) != dim:
            raise IncompatibleEmbeddingModelError(
                f"Provider dimension {dim} is incompatible with indexed dimension "
                f"{stored['vector_dimension']}. Rebuild the embedding index before searching."
            )

    def _get_vector(self, content_hash: str, content: str) -> list[float]:
        """Return embedding vector, reading from persistent cache or computing on miss."""
        model_name = getattr(self._provider, "model_name", None)
        if model_name:
            get_vector = getattr(self._store, "get_vector", None)
            if callable(get_vector):
                cached = get_vector(content_hash, model_name)
                if cached is not None:
                    return cached

        vec = self._provider.embed(content)

        if model_name:
            set_vector = getattr(self._store, "set_vector", None)
            if callable(set_vector):
                set_vector(content_hash, model_name, vec)

        return vec

    def trigger_reindexing(
        self,
        node_path: str | None = None,
        new_provider: EmbeddingProvider | None = None,
    ) -> None:
        """Warm in-memory cache (and persistent cache) for all active chunks.

        If *new_provider* is given the old model's cached vectors are purged first,
        the embedding schema is updated, and the provider is replaced before
        re-embedding.
        """
        if new_provider is not None:
            old_model = getattr(self._provider, "model_name", None)
            if old_model:
                delete_vectors = getattr(self._store, "delete_vectors_for_model", None)
                if callable(delete_vectors):
                    delete_vectors(old_model)
            self._provider = new_provider
            set_schema = getattr(self._store, "set_embedding_schema", None)
            if callable(set_schema):
                new_model = getattr(new_provider, "model_name", None)
                new_dim = getattr(new_provider, "embed_dimension", -1)
                if new_model:
                    set_schema(new_model, new_dim)

        for chunk in self._store.list_chunks():  # type: ignore[attr-defined]
            if chunk.status == "active":
                self._get_vector(chunk.content_hash, chunk.content)

    def search(
        self,
        query: str,
        *,
        root_path: str | None = None,
        limit: int = 10,
        include_stale: bool = False,
        threshold: float = 0.0,
    ) -> list[SearchResult]:
        if not query.strip() or limit <= 0:
            return []

        query_vec = self._provider.embed(query)
        results: list[SearchResult] = []

        for chunk in self._store.list_chunks():  # type: ignore[attr-defined]
            if not include_stale and chunk.status != "active":
                continue
            if not _path_in_scope(chunk.node_path, root_path):
                continue

            score = cosine_similarity(query_vec, self._get_vector(chunk.content_hash, chunk.content))
            if score <= threshold:
                continue

            text = chunk.content
            snippet = text[:157].rstrip() + "..." if len(text) > 160 else text
            results.append(
                SearchResult(
                    path=chunk.node_path,
                    source="chunk:semantic",
                    score=score,
                    snippet=snippet,
                    chunk_id=chunk.id,
                    layer=chunk.layer,
                    content_type=chunk.content_type,
                    status=chunk.status,
                )
            )

        results.sort(key=lambda r: (-r.score, r.path))
        return results[:limit]
