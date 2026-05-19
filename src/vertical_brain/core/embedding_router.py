from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

from vertical_brain.core.embedding_search import EmbeddingSearch
from vertical_brain.core.gold import gold_aspect_embed_key, normalize_gold_aspect_text, parse_gold_content
from vertical_brain.core.models import Chunk, EmbeddingRouteCandidate
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
        *,
        chunk_filter: Callable[[Chunk], bool] | None = None,
    ) -> list[EmbeddingRouteCandidate]:
        query_vec = self._provider.embed(text)
        expected_dimension = len(query_vec)
        query_tokens = _path_tokens(text)

        # --- semantic scoring against active Gold aspects ---
        gold_chunks = [
            c for c in self._store.list_chunks()  # type: ignore[attr-defined]
            if c.layer == "gold" and c.status == "active"
        ]
        if chunk_filter is not None:
            gold_chunks = [c for c in gold_chunks if chunk_filter(c)]
        gold_overflow_parents = _gold_overflow_parents(self._store)
        gold_overflow_paths = set(gold_overflow_parents)

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
                path = _canonical_gold_path(chunk.node_path, gold_overflow_parents)
                if path not in best or sim > best[path][0]:
                    best[path] = (sim, aspect_text)

        # --- path fallback for nodes with no Gold chunk ---
        gold_paths = set(best.keys())
        for node in self._store.list_nodes():  # type: ignore[attr-defined]
            if node.path in gold_paths or node.path in gold_overflow_paths:
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

    def find_candidates_with_fallback(
        self,
        text: str,
        *,
        threshold: float = 0.0,
        limit: int = 5,
        fallback_threshold: float = 0.55,
        chunk_filter: Callable[[Chunk], bool] | None = None,
    ) -> list[EmbeddingRouteCandidate]:
        """Gold-first routing with a Bronze/Silver semantic fallback.

        When the best Gold candidate scores below ``fallback_threshold`` (default
        0.55, tuned so confident Gold-routed queries skip the extra work), this
        method runs a semantic search over Bronze and Silver chunks and merges
        the resulting namespaces with the Gold candidates. Each candidate carries
        ``match_source`` so callers can distinguish a Gold hit from a content
        rescue.

        This costs at most one extra EmbeddingSearch pass (which itself relies on
        cached chunk vectors after a reindex). When Gold is confident, this method
        behaves identically to ``find_candidates``.

        ``chunk_filter`` (v1.11) is applied both to the Gold-aspect scan and to
        the Bronze/Silver fallback sweep so an ACL hides a namespace consistently
        on both branches.
        """
        gold = self.find_candidates(
            text,
            threshold=threshold,
            limit=max(limit * 2, limit),
            chunk_filter=chunk_filter,
        )
        top_score = gold[0].score if gold else -1.0
        if top_score >= fallback_threshold:
            return gold[:limit]

        # Gold did not produce a confident match — consult Bronze/Silver.
        search = EmbeddingSearch(self._store, self._provider)
        fallback_hits = search.search(
            text,
            limit=max(limit * 3, limit),
            threshold=threshold,
            chunk_filter=chunk_filter,
        )
        fallback_by_path: dict[str, tuple[float, str]] = {}
        for hit in fallback_hits:
            if hit.layer == "gold":
                # Gold was already considered upstream; avoid double-counting.
                continue
            existing = fallback_by_path.get(hit.path)
            if existing is None or hit.score > existing[0]:
                fallback_by_path[hit.path] = (hit.score, hit.snippet)

        merged: dict[str, EmbeddingRouteCandidate] = {
            c.path: c for c in gold
        }
        for path, (score, snippet) in fallback_by_path.items():
            current = merged.get(path)
            if current is None or score > current.score:
                merged[path] = EmbeddingRouteCandidate(
                    path=path,
                    score=score,
                    gold_summary=snippet,
                    match_source="content_fallback",
                )

        ordered = sorted(merged.values(), key=lambda c: (-c.score, c.path))
        return ordered[:limit]

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


def _gold_overflow_parents(store: "StorageProvider") -> dict[str, str]:
    return {
        link.target_path: link.source_path
        for link in store.list_links()
        if link.link_type == "gold_overflow"
    }


def _canonical_gold_path(path: str, overflow_parents: dict[str, str]) -> str:
    current = path
    seen = {current}
    while current in overflow_parents:
        parent = overflow_parents[current]
        if parent in seen:
            return path
        seen.add(parent)
        current = parent
    return current


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
