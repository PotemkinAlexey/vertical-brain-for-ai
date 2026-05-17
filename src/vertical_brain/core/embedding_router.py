from __future__ import annotations

import re
from typing import TYPE_CHECKING

from vertical_brain.core.gold import gold_aspect_embed_key, normalize_gold_aspect_text, parse_gold_content
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
    """Routes text to namespaces by comparing against individual Gold aspects.

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
        expected_dimension = len(query_vec)
        query_tokens = _path_tokens(text)

        # --- semantic scoring against active Gold aspects ---
        gold_chunks = [
            c for c in self._store.list_chunks()  # type: ignore[attr-defined]
            if c.layer == "gold" and c.status == "active"
        ]

        best: dict[str, tuple[float, str]] = {}
        for chunk in gold_chunks:
            # Gold is an index: each aspect is a short independent routing anchor.
            # Embedding the joined chunk would average unrelated topics together.
            for aspect_text in _gold_aspect_texts(chunk.content):
                aspect_vec = self._get_aspect_vector(
                    aspect_text,
                    expected_dimension=expected_dimension,
                )
                sim = cosine_similarity(query_vec, aspect_vec)
                path = chunk.node_path
                if path not in best or sim > best[path][0]:
                    best[path] = (sim, aspect_text)

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

    def _get_aspect_vector(self, aspect_text: str, *, expected_dimension: int) -> list[float]:
        embed_text = normalize_gold_aspect_text(aspect_text)
        cache_key = gold_aspect_embed_key(embed_text)
        model_name = getattr(self._provider, "model_name", None)

        if model_name:
            get_vector = getattr(self._store, "get_vector", None)
            if callable(get_vector):
                cached = _coerce_vector(
                    get_vector(cache_key, model_name),
                    expected_dimension=expected_dimension,
                )
                if cached is not None:
                    return cached
        elif cache_key in self._cache:
            cached = _coerce_vector(self._cache[cache_key], expected_dimension=expected_dimension)
            if cached is not None:
                return cached

        vector = _coerce_vector(
            self._provider.embed(embed_text),
            expected_dimension=expected_dimension,
        )
        if vector is None:
            raise ValueError("Embedding provider returned an invalid vector")

        if model_name:
            set_vector = getattr(self._store, "set_vector", None)
            if callable(set_vector):
                set_vector(cache_key, model_name, vector)
        else:
            self._cache[cache_key] = vector

        return vector


def _gold_aspect_texts(content: str) -> list[str]:
    return [normalize_gold_aspect_text(text) for text in parse_gold_content(content) if text.strip()]


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
