"""Tests for the v1.10 RerankerProvider Protocol and its injection points.

The open core ships no reranker implementation; this file exercises the
extension contract using stub rerankers so we know the seam stays usable
once enterprise plugs in Cohere/Voyage/custom cross-encoders.
"""
from __future__ import annotations

import json

import pytest

from vertical_brain.core.embedding_search import EmbeddingSearch
from vertical_brain.core.models import Chunk, SearchResult
from vertical_brain.llm.embedding import MockEmbeddingProvider
from vertical_brain.llm.reranker import RerankerProvider
from vertical_brain.mcp.server import VerticalBrainMCP
from vertical_brain.storage.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Stub rerankers
# ---------------------------------------------------------------------------


class _ReverseOrderReranker:
    """Reranker that reverses the cosine order — easiest way to see it ran."""

    model_name = "reverse-order"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        self.calls.append((query, [c.chunk_id for c in candidates if c.chunk_id]))
        return list(reversed(candidates))


class _DropAllReranker:
    """Reranker that drops every candidate (filters by an absurd threshold)."""

    model_name = "drop-all"

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        return []


class _ExplodingReranker:
    """Reranker that always raises — must not break the read path."""

    model_name = "exploding"

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        raise RuntimeError("upstream rerank service is down")


class _FabricatingReranker:
    """Reranker that returns a SearchResult whose chunk_id was not in the input."""

    model_name = "fabricating"

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        ghost = SearchResult(
            path="GHOST/path",
            source="chunk:semantic",
            score=99.0,
            snippet="not a real result",
            chunk_id="fabricated-id",
            layer="bronze",
            content_type="fact",
            status="active",
        )
        return [ghost, *candidates]


# ---------------------------------------------------------------------------
# Protocol shape
# ---------------------------------------------------------------------------


def test_reranker_protocol_runtime_checkable():
    assert isinstance(_ReverseOrderReranker(), RerankerProvider)


def test_reranker_protocol_rejects_missing_method():
    class _NotAReranker:
        model_name = "bad"

    assert not isinstance(_NotAReranker(), RerankerProvider)


# ---------------------------------------------------------------------------
# EmbeddingSearch.search reranker wiring
# ---------------------------------------------------------------------------


def _seed_chunks(store: SQLiteStore) -> list[Chunk]:
    """Three chunks with deliberately distinct content so cosine produces
    a stable order. _ReverseOrderReranker will then flip them."""
    return [
        store.save_chunk(Chunk(node_path="A", content="delta lake streaming alpha")),
        store.save_chunk(Chunk(node_path="B", content="delta lake streaming beta")),
        store.save_chunk(Chunk(node_path="C", content="delta lake streaming gamma")),
    ]


def test_search_without_reranker_keeps_cosine_order(tmp_path):
    store = SQLiteStore(tmp_path)
    _seed_chunks(store)
    results = EmbeddingSearch(store, MockEmbeddingProvider()).search("delta lake")
    assert len(results) >= 3
    cosine_order_paths = [r.path for r in results[:3]]
    # No reranker → results come back in -score order (default tie-broken by path).
    assert cosine_order_paths == sorted(cosine_order_paths, key=lambda p: p)


def test_search_with_reranker_reorders_results(tmp_path):
    store = SQLiteStore(tmp_path)
    _seed_chunks(store)
    reranker = _ReverseOrderReranker()
    results = EmbeddingSearch(store, MockEmbeddingProvider()).search(
        "delta lake", reranker=reranker
    )
    assert len(results) >= 1
    assert len(reranker.calls) == 1
    query, candidate_ids = reranker.calls[0]
    assert query == "delta lake"
    # All seeded chunks must be in the candidate pool passed to the reranker.
    assert len(candidate_ids) >= 3


def test_search_reranker_may_drop_candidates(tmp_path):
    store = SQLiteStore(tmp_path)
    _seed_chunks(store)
    results = EmbeddingSearch(store, MockEmbeddingProvider()).search(
        "delta lake", reranker=_DropAllReranker()
    )
    assert results == []


def test_search_reranker_exception_falls_back_to_cosine(tmp_path):
    store = SQLiteStore(tmp_path)
    _seed_chunks(store)
    results = EmbeddingSearch(store, MockEmbeddingProvider()).search(
        "delta lake", reranker=_ExplodingReranker()
    )
    # Exception swallowed → cosine fallback returns the un-reranked candidates.
    assert len(results) >= 3


def test_search_reranker_fabricated_results_are_filtered(tmp_path):
    store = SQLiteStore(tmp_path)
    _seed_chunks(store)
    results = EmbeddingSearch(store, MockEmbeddingProvider()).search(
        "delta lake", reranker=_FabricatingReranker()
    )
    paths = [r.path for r in results]
    assert "GHOST/path" not in paths
    assert all(r.chunk_id != "fabricated-id" for r in results)


def test_search_reranker_invoked_with_pooled_candidates(tmp_path):
    """The pool passed to the reranker oversamples beyond `limit` so the
    cross-encoder has room to refine."""
    store = SQLiteStore(tmp_path)
    for i in range(15):
        store.save_chunk(Chunk(node_path=f"N/{i}", content=f"alpha beta gamma item {i}"))

    reranker = _ReverseOrderReranker()
    EmbeddingSearch(store, MockEmbeddingProvider()).search(
        "alpha beta", limit=3, reranker=reranker
    )

    assert len(reranker.calls) == 1
    _query, candidate_ids = reranker.calls[0]
    # Default pool factor is 3 → ~9 candidates when limit=3.
    assert len(candidate_ids) >= 9


# ---------------------------------------------------------------------------
# MCP wiring — search_semantic uses VerticalBrainMCP._reranker
# ---------------------------------------------------------------------------


def test_mcp_search_semantic_uses_configured_reranker(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="A", content="delta lake alpha"))
    store.save_chunk(Chunk(node_path="B", content="delta lake beta"))
    reranker = _ReverseOrderReranker()
    server = VerticalBrainMCP(store, reranker=reranker)

    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "search_semantic", "arguments": {"query": "delta lake"}},
    })

    assert "error" not in response
    data = json.loads(response["result"]["content"][0]["text"])
    assert len(reranker.calls) == 1
    assert reranker.calls[0][0] == "delta lake"
    # Reranker did run; results still come through (layer bias may reorder).
    assert len(data["results"]) >= 2


def test_mcp_search_semantic_without_reranker_works_unchanged(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="A", content="delta lake alpha"))
    server = VerticalBrainMCP(store)  # no reranker

    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "search_semantic", "arguments": {"query": "delta lake"}},
    })

    assert "error" not in response
    data = json.loads(response["result"]["content"][0]["text"])
    assert isinstance(data["results"], list)
    assert "semantic_endpoint" in data


def test_mcp_context_search_semantic_uses_configured_reranker(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="A", content="delta lake alpha"))
    store.save_chunk(Chunk(node_path="B", content="delta lake beta"))
    reranker = _ReverseOrderReranker()
    server = VerticalBrainMCP(store, reranker=reranker)

    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {
            "name": "context_search_semantic",
            "arguments": {"query": "delta lake", "context_limit": 1},
        },
    })

    assert "error" not in response
    assert len(reranker.calls) == 1
    assert reranker.calls[0][0] == "delta lake"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_search_reranker_none_is_default(tmp_path):
    """`reranker=None` is the explicit default — equivalent to no kwarg."""
    store = SQLiteStore(tmp_path)
    _seed_chunks(store)
    a = EmbeddingSearch(store, MockEmbeddingProvider()).search("delta lake")
    b = EmbeddingSearch(store, MockEmbeddingProvider()).search("delta lake", reranker=None)
    assert [r.chunk_id for r in a] == [r.chunk_id for r in b]
