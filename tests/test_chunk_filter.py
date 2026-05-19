"""Tests for the v1.11 `chunk_filter` ACL seam.

`chunk_filter` is an optional callable `Callable[[Chunk], bool]` accepted by
all read-path APIs:

- ContextLock.open_locked_context
- BrainSearch.search
- EmbeddingSearch.search
- EmbeddingRouter.find_candidates / find_candidates_with_fallback
- ContextSession.search_locked_context / search_locked_context_semantic
- VerticalBrainMCP._current_chunk_filter() (per-request hook)

These tests exercise the seam end-to-end without depending on any
particular ACL implementation — enterprise drops in `chunk.metadata`,
tenant_id, classification, etc. once v1.12 lands.
"""
from __future__ import annotations

import json

from vertical_brain.core.context_lock import ContextLock
from vertical_brain.core.context_session import ContextSession
from vertical_brain.core.embedding_router import EmbeddingRouter
from vertical_brain.core.embedding_search import EmbeddingSearch
from vertical_brain.core.models import Chunk, ContextBudget, ContextPolicy
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.core.search import BrainSearch
from vertical_brain.llm.embedding import MockEmbeddingProvider
from vertical_brain.mcp.server import VerticalBrainMCP
from vertical_brain.storage.sqlite_store import SQLiteStore


# A trivial ACL: hide everything sourced from "secret".
def _hide_secret(chunk: Chunk) -> bool:
    return chunk.source != "secret"


# ---------------------------------------------------------------------------
# ContextLock
# ---------------------------------------------------------------------------


def test_context_lock_filter_hides_target_bronze(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="public fact", source="model"))
    store.save_chunk(Chunk(node_path="WORK/A", content="secret fact", source="secret"))

    lock = ContextLock(store)
    locked = lock.open_locked_context(
        "WORK/A",
        policy=ContextPolicy(),
        budget=ContextBudget(),
        chunk_filter=_hide_secret,
    )
    contents = [item.content for item in locked.items]
    assert "public fact" in contents
    assert "secret fact" not in contents


def test_context_lock_filter_hides_ancestor_gold(tmp_path):
    store = SQLiteStore(tmp_path)
    ex = StorageOperationExecutor(store)
    from vertical_brain.core.models import (
        ChunkInput,
        StorageOperation,
        StorageOperationBatch,
    )
    # Seed Silver + Gold at the ancestor, then a target chunk one level down.
    ex.apply(StorageOperation(
        operation="append_chunk",
        target_path="WORK",
        chunk=ChunkInput(content="parent silver", layer="silver"),
    ))
    ex.apply_batch(StorageOperationBatch(operations=[
        StorageOperation(
            operation="append_gold_aspect",
            target_path="WORK",
            gold_aspect="work routing tag",
        ),
    ]))
    # Mark the Gold chunk's source as "secret" so the ACL hides it.
    gold = [c for c in store.get_chunks_by_path("WORK") if c.layer == "gold"][0]
    gold.source = "secret"
    store.update_chunk(gold)

    store.save_chunk(Chunk(node_path="WORK/A", content="child bronze"))
    locked = ContextLock(store).open_locked_context(
        "WORK/A",
        policy=ContextPolicy(include_ancestors=True),
        budget=ContextBudget(),
        chunk_filter=_hide_secret,
    )
    # Ancestor Gold is hidden; child Bronze is visible.
    ancestor_layers = [item.layer for item in locked.items if item.path == "WORK"]
    assert "gold" not in ancestor_layers


def test_context_lock_filter_none_is_default(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="a", source="model"))
    store.save_chunk(Chunk(node_path="WORK/A", content="b", source="secret"))
    locked = ContextLock(store).open_locked_context("WORK/A")
    contents = {item.content for item in locked.items}
    assert {"a", "b"}.issubset(contents)


# ---------------------------------------------------------------------------
# EmbeddingSearch
# ---------------------------------------------------------------------------


def test_embedding_search_filter_hides_results(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="delta lake public", source="model"))
    store.save_chunk(Chunk(node_path="WORK/B", content="delta lake secret", source="secret"))

    results = EmbeddingSearch(store, MockEmbeddingProvider()).search(
        "delta lake", chunk_filter=_hide_secret
    )
    paths = [r.path for r in results]
    assert "WORK/A" in paths
    assert "WORK/B" not in paths


def test_embedding_search_filter_with_reranker(tmp_path):
    """chunk_filter applies before the reranker even sees a candidate."""
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="delta lake public", source="model"))
    store.save_chunk(Chunk(node_path="WORK/B", content="delta lake secret", source="secret"))

    class _CaptureReranker:
        model_name = "capture"
        def __init__(self):
            self.seen_paths: list[str] = []
        def rerank(self, query, candidates):
            self.seen_paths = [c.path for c in candidates]
            return candidates

    reranker = _CaptureReranker()
    EmbeddingSearch(store, MockEmbeddingProvider()).search(
        "delta lake", reranker=reranker, chunk_filter=_hide_secret
    )
    assert "WORK/A" in reranker.seen_paths
    assert "WORK/B" not in reranker.seen_paths


# ---------------------------------------------------------------------------
# BrainSearch (lexical FTS)
# ---------------------------------------------------------------------------


def test_brain_search_filter_hides_results(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="delta lake public", source="model"))
    store.save_chunk(Chunk(node_path="WORK/B", content="delta lake secret", source="secret"))

    results = BrainSearch(store).search("delta lake", chunk_filter=_hide_secret)
    paths = [r.path for r in results]
    assert "WORK/A" in paths
    assert "WORK/B" not in paths


def test_brain_search_filter_none_is_default(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="alpha", source="model"))
    store.save_chunk(Chunk(node_path="WORK/B", content="alpha", source="secret"))

    a = BrainSearch(store).search("alpha")
    b = BrainSearch(store).search("alpha", chunk_filter=None)
    assert sorted(r.path for r in a) == sorted(r.path for r in b)


# ---------------------------------------------------------------------------
# EmbeddingRouter
# ---------------------------------------------------------------------------


def test_embedding_router_filter_hides_gold_chunks(tmp_path):
    store = SQLiteStore(tmp_path)
    ex = StorageOperationExecutor(store)
    from vertical_brain.core.models import (
        ChunkInput,
        StorageOperation,
        StorageOperationBatch,
    )
    # Seed Silver at both, then Gold at both, then mark one Gold chunk as secret.
    for path, label in [("WORK/Public", "public routing tag"), ("WORK/Secret", "secret routing tag")]:
        ex.apply(StorageOperation(
            operation="append_chunk",
            target_path=path,
            chunk=ChunkInput(content=f"silver for {path}", layer="silver"),
        ))
        ex.apply_batch(StorageOperationBatch(operations=[
            StorageOperation(
                operation="append_gold_aspect",
                target_path=path,
                gold_aspect=label,
            ),
        ]))

    # Mark the WORK/Secret Gold chunk as source="secret".
    for chunk in store.get_chunks_by_path("WORK/Secret"):
        if chunk.layer == "gold":
            chunk.source = "secret"
            store.update_chunk(chunk)

    router = EmbeddingRouter(store, MockEmbeddingProvider())
    candidates = router.find_candidates("public routing tag", chunk_filter=_hide_secret)
    paths = {c.path for c in candidates}
    assert "WORK/Public" in paths
    # WORK/Secret's Gold is hidden; it may still appear via the path-token
    # fallback, but its score should be ≤ the path cap 0.45.
    for cand in candidates:
        if cand.path == "WORK/Secret":
            assert cand.score <= 0.45


# ---------------------------------------------------------------------------
# VerticalBrainMCP — _current_chunk_filter() hook
# ---------------------------------------------------------------------------


class _ACLServer(VerticalBrainMCP):
    """Enterprise-style subclass injecting an ACL filter for the whole request."""

    def _current_chunk_filter(self):
        return _hide_secret


def test_mcp_default_current_chunk_filter_is_none(tmp_path):
    store = SQLiteStore(tmp_path)
    server = VerticalBrainMCP(store)
    assert server._current_chunk_filter() is None


def test_mcp_subclass_chunk_filter_hides_results_in_search(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="delta lake public", source="model"))
    store.save_chunk(Chunk(node_path="WORK/B", content="delta lake secret", source="secret"))

    server = _ACLServer(store)
    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "search", "arguments": {"query": "delta lake"}},
    })

    data = json.loads(response["result"]["content"][0]["text"])
    paths = {r["path"] for r in data}
    assert "WORK/A" in paths
    assert "WORK/B" not in paths


def test_mcp_subclass_chunk_filter_hides_in_search_semantic(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="delta lake public", source="model"))
    store.save_chunk(Chunk(node_path="WORK/B", content="delta lake secret", source="secret"))

    server = _ACLServer(store)
    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "search_semantic", "arguments": {"query": "delta lake"}},
    })

    data = json.loads(response["result"]["content"][0]["text"])
    paths = {r["path"] for r in data["results"]}
    assert "WORK/B" not in paths


def test_mcp_subclass_chunk_filter_hides_in_read_context(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="public fact", source="model"))
    store.save_chunk(Chunk(node_path="WORK/A", content="secret fact", source="secret"))

    server = _ACLServer(store)
    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "read_context", "arguments": {"path": "WORK/A"}},
    })

    data = json.loads(response["result"]["content"][0]["text"])
    contents = " ".join(item["content"] for item in data["items"])
    assert "public fact" in contents
    assert "secret fact" not in contents


def test_mcp_subclass_chunk_filter_hides_in_context_search(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="delta lake public", source="model"))
    store.save_chunk(Chunk(node_path="WORK/B", content="delta lake secret", source="secret"))

    server = _ACLServer(store)
    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "context_search", "arguments": {"query": "delta lake"}},
    })

    data = json.loads(response["result"]["content"][0]["text"])
    handle_paths = {h["path"] for h in data["candidate_handles"]}
    assert "WORK/A" in handle_paths
    assert "WORK/B" not in handle_paths
