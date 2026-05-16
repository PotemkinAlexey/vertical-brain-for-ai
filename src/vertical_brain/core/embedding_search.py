from __future__ import annotations

from typing import TYPE_CHECKING

from vertical_brain.core.models import SearchResult
from vertical_brain.core.search import _path_in_scope
from vertical_brain.llm.embedding import EmbeddingProvider, cosine_similarity

if TYPE_CHECKING:
    from vertical_brain.storage.protocol import StorageProvider


class EmbeddingSearch:
    """Semantic chunk search using cosine similarity against embedded chunk content."""

    def __init__(self, store: "StorageProvider", provider: EmbeddingProvider) -> None:
        self._store = store
        self._provider = provider
        self._cache: dict[str, list[float]] = {}

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
