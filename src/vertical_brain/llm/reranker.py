"""Reranker provider Protocol — v1.10 extension point.

Vertical Brain v1.10 introduces an optional reranker stage that sits
between cosine retrieval (`EmbeddingSearch.search`) and layer biasing
(`apply_layer_bias`). The Protocol is defined here so enterprise
implementations (Cohere Rerank, Voyage Rerank, cross-encoder via
sentence-transformers, in-house models) can drop in without further
changes to the open core.

There is intentionally **no default implementation** in the open core:
the read-path remains cosine + layer-bias when no reranker is wired,
which is dependency-free and provider-agnostic. When a `RerankerProvider`
is supplied to `VerticalBrainMCP` (or passed directly to
`EmbeddingSearch.search` / `ContextSession.search_locked_context_semantic`)
it is applied to the candidate list before layer biasing.

Contract:

- `rerank(query, candidates)` returns a possibly-reordered, possibly-
  shorter `list[SearchResult]`. Each output element MUST be a
  `SearchResult` that was present in the input (same `chunk_id`,
  `path`, `layer`, etc.) — only `score` may be refreshed.
- The implementation MAY drop low-relevance candidates (apply its own
  internal threshold) but MUST NOT invent new ones; fabricated results
  are filtered out by the caller as a safety net.
- Order returned defines the post-rerank ranking; subsequent layer
  biasing will refine the order within each (layer, content_type)
  bucket but will not drop results.
- Exceptions raised by `rerank` must not break the read path —
  callers swallow them and fall back to the pre-rerank cosine order.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vertical_brain.core.models import SearchResult


@runtime_checkable
class RerankerProvider(Protocol):
    """Cross-encoder / re-scoring stage applied after cosine retrieval."""

    model_name: str

    def rerank(
        self,
        query: str,
        candidates: list["SearchResult"],
    ) -> list["SearchResult"]: ...
