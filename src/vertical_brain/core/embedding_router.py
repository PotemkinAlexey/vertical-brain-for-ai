from __future__ import annotations

from typing import TYPE_CHECKING

from vertical_brain.core.models import EmbeddingRouteCandidate
from vertical_brain.llm.embedding import EmbeddingProvider, cosine_similarity

if TYPE_CHECKING:
    from vertical_brain.storage.protocol import StorageProvider


class EmbeddingRouter:
    """Routes text to namespaces by comparing against Gold chunk embeddings."""

    def __init__(self, store: "StorageProvider", provider: EmbeddingProvider) -> None:
        self._store = store
        self._provider = provider
        self._cache: dict[str, list[float]] = {}

    def find_candidates(
        self,
        text: str,
        threshold: float = 0.0,
        limit: int = 5,
    ) -> list[EmbeddingRouteCandidate]:
        query_vec = self._provider.embed(text)
        gold_chunks = [
            c for c in self._store.list_chunks()  # type: ignore[attr-defined]
            if c.layer == "gold" and c.status == "active"
        ]

        best: dict[str, tuple[float, str]] = {}
        for chunk in gold_chunks:
            if chunk.content not in self._cache:
                self._cache[chunk.content] = self._provider.embed(chunk.content)
            sim = cosine_similarity(query_vec, self._cache[chunk.content])
            path = chunk.node_path
            if path not in best or sim > best[path][0]:
                best[path] = (sim, chunk.content)

        candidates = [
            EmbeddingRouteCandidate(path=path, score=score, gold_summary=summary)
            for path, (score, summary) in best.items()
            if score >= threshold
        ]
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:limit]
