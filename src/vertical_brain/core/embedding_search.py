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
        self._cache: dict[str, list[float]] = {}
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

    def trigger_reindexing(
        self,
        node_path: str | None = None,
        new_provider: EmbeddingProvider | None = None,
    ) -> None:
        """Schedule reindexing under a new embedding model. Not yet implemented."""
        raise NotImplementedError(
            "Reindexing is not yet implemented. To migrate to a new embedding model, "
            "clear the embedding schema and rebuild the index with the new provider."
        )

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

            if chunk.content not in self._cache:
                self._cache[chunk.content] = self._provider.embed(chunk.content)
            score = cosine_similarity(query_vec, self._cache[chunk.content])
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
