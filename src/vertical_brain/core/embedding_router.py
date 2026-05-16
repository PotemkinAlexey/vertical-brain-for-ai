from __future__ import annotations

import re
from typing import TYPE_CHECKING

from vertical_brain.core.models import EmbeddingRouteCandidate
from vertical_brain.llm.embedding import EmbeddingProvider, cosine_similarity

if TYPE_CHECKING:
    from vertical_brain.storage.protocol import StorageProvider

_PATH_MATCH_CAP = 0.45

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _expand_segment(seg: str) -> set[str]:
    """Return all token forms for a single path segment.

    "AutoLoader"      → {"autoloader", "auto", "loader"}
    "StructuredStreaming" → {"structuredstreaming", "structured", "streaming"}
    "my-namespace"    → {"my-namespace", "my", "namespace"}
    """
    whole = seg.lower()
    tokens: set[str] = {whole}
    # camelCase / PascalCase split
    parts = _CAMEL_BOUNDARY.sub(" ", seg).split()
    # also split on hyphen/underscore/space
    for part in parts:
        for sub in re.split(r"[-_ ]+", part):
            if sub:
                tokens.add(sub.lower())
    return tokens


def _path_tokens(path: str) -> set[str]:
    """Expand all segments of a slash-separated namespace path."""
    tokens: set[str] = set()
    for seg in path.split("/"):
        if seg:
            tokens |= _expand_segment(seg)
    return tokens


def _path_score(query_tokens: set[str], path: str) -> float:
    """Normalized overlap between expanded query tokens and expanded path tokens."""
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & _path_tokens(path))
    return min(_PATH_MATCH_CAP, overlap / len(query_tokens))


class EmbeddingRouter:
    """Routes text to namespaces by comparing against Gold chunk embeddings.

    Fallback: namespaces with no Gold chunk can still appear as candidates
    based on lexical path/name overlap with the query.  Semantic Gold scores
    always outrank pure path matches because the path cap is 0.45.
    """

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
        query_tokens = _path_tokens(text)

        # --- semantic scoring against active Gold chunks ---
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

        # --- path fallback for nodes with no Gold chunk ---
        gold_paths = set(best.keys())
        for node in self._store.list_nodes():  # type: ignore[attr-defined]
            if node.path in gold_paths:
                continue
            score = _path_score(query_tokens, node.path)
            if score > 0:
                best[node.path] = (score, "")

        candidates = [
            EmbeddingRouteCandidate(path=path, score=score, gold_summary=summary)
            for path, (score, summary) in best.items()
            if score >= threshold
        ]
        candidates.sort(key=lambda c: (-c.score, c.path))
        return candidates[:limit]
