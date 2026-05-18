from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from vertical_brain.core.gold import gold_embed_text
from vertical_brain.core.models import SearchResult
from vertical_brain.core.namespace_map import normalize_namespace_root_path
from vertical_brain.llm.embedding import EmbeddingProvider, cosine_similarity

if TYPE_CHECKING:
    from vertical_brain.core.models import Chunk
    from vertical_brain.storage.protocol import StorageProvider


def _chunk_embed_info(chunk: "Chunk") -> tuple[str, str]:
    """Return ``(cache_key, embed_text)`` for a chunk.

    Gold chunks are embedded as clean joined aspect text (no JSON noise).
    The cache key is derived from the embed text so that if the Gold content
    changes, the old vector is evicted naturally (different key → cache miss).

    All other layers use the chunk's own ``content_hash`` and ``content``.
    """
    if chunk.layer == "gold":
        text = gold_embed_text(chunk.content)
        key = hashlib.sha256(text.encode()).hexdigest()
        return key, text
    return chunk.content_hash, chunk.content


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

    def _get_vector(
        self,
        content_hash: str,
        content: str,
        *,
        expected_dimension: int | None = None,
    ) -> list[float]:
        """Return embedding vector, reading from persistent cache or computing on miss."""
        model_name = getattr(self._provider, "model_name", None)
        expected_dimension = self._expected_dimension(expected_dimension)
        if model_name:
            get_vector = getattr(self._store, "get_vector", None)
            if callable(get_vector):
                cached = get_vector(content_hash, model_name)
                cached_vector = _coerce_vector(cached, expected_dimension=expected_dimension)
                if cached_vector is not None:
                    return cached_vector

        vec = self._embed(content, expected_dimension=expected_dimension)

        if model_name:
            set_vector = getattr(self._store, "set_vector", None)
            if callable(set_vector):
                set_vector(content_hash, model_name, vec)

        return vec

    def _embed(self, text: str, *, expected_dimension: int | None = None) -> list[float]:
        vector = _coerce_vector(
            self._provider.embed(text),
            expected_dimension=self._expected_dimension(expected_dimension),
        )
        if vector is None:
            raise ValueError("Embedding provider returned an invalid vector")
        return vector

    def _expected_dimension(self, fallback: int | None = None) -> int | None:
        provider_dim = getattr(self._provider, "embed_dimension", -1)
        if isinstance(provider_dim, int) and not isinstance(provider_dim, bool) and provider_dim > 0:
            return provider_dim
        return fallback

    def _candidate_chunks(
        self,
        *,
        root_path: str | None,
        include_stale: bool,
    ) -> list["Chunk"]:
        root_path = normalize_namespace_root_path(root_path)
        if root_path:
            chunks = self._store.get_chunks_by_path(root_path, include_children=True)  # type: ignore[attr-defined]
        else:
            chunks = self._store.list_chunks()  # type: ignore[attr-defined]
        if include_stale:
            return chunks
        return [chunk for chunk in chunks if chunk.status == "active"]

    def trigger_reindexing(
        self,
        new_provider: EmbeddingProvider | None = None,
        *,
        node_path: str | None = None,
    ) -> int:
        """Warm persistent cache for all active chunks; return count of chunks indexed.

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

        count = 0
        for chunk in self._candidate_chunks(root_path=node_path, include_stale=False):
            cache_key, embed_text = _chunk_embed_info(chunk)
            self._get_vector(cache_key, embed_text)
            count += 1
        return count

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

        query_vec = self._embed(query)
        results: list[SearchResult] = []

        for chunk in self._candidate_chunks(root_path=root_path, include_stale=include_stale):
            cache_key, embed_text = _chunk_embed_info(chunk)
            score = cosine_similarity(
                query_vec,
                self._get_vector(
                    cache_key,
                    embed_text,
                    expected_dimension=len(query_vec),
                ),
            )
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


def _coerce_vector(vector: object, *, expected_dimension: int | None = None) -> list[float] | None:
    if not isinstance(vector, list):
        return None
    if expected_dimension is not None and len(vector) != expected_dimension:
        return None
    coerced: list[float] = []
    for value in vector:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        coerced.append(float(value))
    return coerced
