"""Read-path next-step hints for MCP responses.

Embodies the v1.5 contract:

    Gold     = routing (where to look)
    Silver   = current answer
    Bronze   = evidence (cited via chunk_id)
    Vector   = recall assist across chunks (lights up with a real
               embedding endpoint; on Mock it degrades to bag-of-words
               overlap — still usable, just not synonym-aware).

The functions here are pure and provider-aware: when `EmbeddingProvider`
is `MockEmbeddingProvider` we still emit hints (search_semantic remains
the same tool name in either mode), but AGENTS.md notes that real
semantic recall requires a configured endpoint.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from vertical_brain.core.search import tokenize_query
from vertical_brain.llm.embedding import MockEmbeddingProvider, cosine_similarity

if TYPE_CHECKING:
    from vertical_brain.core.models import ContextItem, SearchResult
    from vertical_brain.llm.embedding import EmbeddingProvider


# Lower = higher priority in the assist ranking.
_LAYER_PRIORITY = {"bronze": 0, "silver": 1, "gold": 2}
_BRONZE_CONTENT_PRIORITY = {
    "reference": 0,
    "fact": 1,
    "decision": 2,
    "correction": 3,
}
_SILVER_MIN_INFORMATIVE_CHARS = 120


def is_semantic_provider(provider: "EmbeddingProvider | None") -> bool:
    """Return True when the provider is something other than the in-process Mock.

    The Mock provider is a bag-of-words hash and does not capture meaning, so
    the read-path treats it as the no-endpoint baseline.
    """
    if provider is None:
        return False
    return not isinstance(provider, MockEmbeddingProvider)


def silver_confidence(query: str | None, silver_item: "ContextItem | None") -> str:
    """Return one of: missing, low, weak, ok.

    Heuristic, no embedding calls. The aim is to flag obvious cases where
    the agent should consider recall assist (search_semantic) rather than
    answer from a thin or off-topic Silver.
    """
    if silver_item is None or not silver_item.content:
        return "missing"
    content = silver_item.content.strip()
    if len(content) < _SILVER_MIN_INFORMATIVE_CHARS:
        return "low"
    if query:
        query_terms = set(tokenize_query(query))
        silver_terms = set(tokenize_query(content))
        if query_terms and not (query_terms & silver_terms):
            return "weak"
    return "ok"


def silver_item_for_target(items: Iterable["ContextItem"], target_path: str) -> "ContextItem | None":
    for item in items:
        if item.layer == "silver" and item.path == target_path:
            return item
    return None


# Cosine threshold above which the semantic upgrade promotes a lexical 'weak'
# Silver confidence to 'ok'. Tuned for nomic-embed-text — related-topic chunks
# typically score 0.55-0.85, off-topic chunks 0.1-0.4. 0.5 is conservative
# enough to avoid promoting unrelated Silvers while catching synonyms.
SILVER_SEMANTIC_UPGRADE_THRESHOLD = 0.5


def semantic_silver_upgrade(
    lexical: str,
    query_vec: list[float] | None,
    silver_vec: list[float] | None,
    *,
    threshold: float = SILVER_SEMANTIC_UPGRADE_THRESHOLD,
) -> str:
    """Promote a lexical 'weak' to 'ok' when the query and Silver are semantically close.

    Pure function — no I/O. The caller is responsible for providing the vectors
    (cached embeddings from the store + a fresh `provider.embed(query)`), and
    for skipping this call entirely when no real embedding endpoint is present.

    Only the `weak` path can be upgraded. `missing` (no Silver) and `low` (Silver
    too short) reflect content shortcomings that cosine cannot fix; `ok` is
    already the best outcome.
    """
    if lexical != "weak":
        return lexical
    if not query_vec or not silver_vec:
        return lexical
    if cosine_similarity(query_vec, silver_vec) >= threshold:
        return "ok"
    return lexical


def route_next_hint(candidates: list, semantic: bool) -> str | None:
    """Hint for route → read_context → (search_semantic) flow."""
    if not candidates:
        return (
            "No Gold-based candidates. Try `search_semantic` with the user's query "
            "to find chunks by content, then `read_context` on the top namespaces."
        )
    top = getattr(candidates[0], "path", None)
    target = top or "<top candidate>"
    base = (
        f"Open the top candidate with `read_context(path=\"{target}\")`. "
        "If its Silver does not answer the question, call "
        f"`search_semantic(query=..., root_path=\"{target}\")` for recall assist, "
        "then read the suggested namespace and cite Bronze chunk_ids."
    )
    if not semantic:
        base += (
            " NOTE: no embedding endpoint configured — semantic recall currently "
            "degrades to word overlap; behaviour will improve when an endpoint is set."
        )
    return base


def read_context_next_hint(
    target_path: str,
    confidence: str,
    semantic: bool,
) -> str | None:
    """Hint emitted alongside `read_context` when Silver looks weak."""
    if confidence == "ok":
        return None
    reason = {
        "missing": "no active Silver at this namespace",
        "low": "Silver is too short to be a current answer",
        "weak": "Silver shares no terms with the query",
    }.get(confidence, confidence)
    base = (
        f"Silver confidence is {confidence!r} ({reason}). "
        f"Try `search_semantic(query=..., root_path=\"{target_path}\")` to surface "
        "Bronze evidence chunks, then re-open `read_context` on the suggested namespace."
    )
    if not semantic:
        base += (
            " NOTE: no embedding endpoint configured — semantic recall currently "
            "degrades to word overlap."
        )
    return base


def _result_sort_key(result: "SearchResult") -> tuple[int, int, float]:
    layer = (result.layer or "").lower()
    layer_rank = _LAYER_PRIORITY.get(layer, len(_LAYER_PRIORITY))
    sub_rank = 99
    if layer == "bronze":
        sub_rank = _BRONZE_CONTENT_PRIORITY.get(
            (result.content_type or "").lower(), 50
        )
    elif layer == "silver":
        sub_rank = 0
    # Negative score so higher scores come first within the same bucket.
    return (layer_rank, sub_rank, -float(result.score))


def apply_layer_bias(results: list["SearchResult"]) -> list["SearchResult"]:
    """Reorder semantic results so evidence-bearing chunks float to the top.

    Preference: Bronze (reference > fact > decision > other) > Silver > Gold.
    Score is still respected within each bucket. Does not drop any result.
    """
    return sorted(results, key=_result_sort_key)


def suggested_paths(results: Iterable["SearchResult"], limit: int = 5) -> list[str]:
    """Top unique namespaces, preserving the (biased) result order."""
    seen: set[str] = set()
    out: list[str] = []
    for r in results:
        if r.path in seen:
            continue
        seen.add(r.path)
        out.append(r.path)
        if len(out) >= limit:
            break
    return out
