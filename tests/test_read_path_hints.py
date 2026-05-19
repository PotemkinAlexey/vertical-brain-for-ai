"""Tests for v1.5 read-path recall assist.

Covers two layers:
  * pure helpers in `vertical_brain.mcp.read_path` (deterministic, no I/O)
  * MCP integration: route / read_context / search_semantic responses
    on the default Mock embedding provider.

The "semantic endpoint configured" branch is exercised by stubbing the
provider with a non-Mock subclass — no Ollama process required.
"""
from __future__ import annotations

import json

import pytest

from vertical_brain.core.models import Chunk, ContextItem, SearchResult
from vertical_brain.llm.embedding import EmbeddingProvider, MockEmbeddingProvider
from vertical_brain.mcp.read_path import (
    apply_layer_bias,
    is_semantic_provider,
    read_context_next_hint,
    route_next_hint,
    semantic_silver_upgrade,
    silver_confidence,
    silver_item_for_target,
    suggested_paths,
)
from vertical_brain.mcp.server import VerticalBrainMCP
from vertical_brain.storage.json_store import JsonStore


# ── pure helpers ────────────────────────────────────────────────────────


def _silver_item(content: str, path: str = "PROJECTS/foo") -> ContextItem:
    return ContextItem(path=path, layer="silver", content=content)


def test_silver_confidence_missing_when_no_item():
    assert silver_confidence("anything", None) == "missing"


def test_silver_confidence_missing_when_empty_content():
    assert silver_confidence("anything", _silver_item("")) == "missing"


def test_silver_confidence_low_when_too_short():
    assert silver_confidence("anything", _silver_item("short summary.")) == "low"


def test_silver_confidence_weak_when_no_overlap():
    content = "This is a long enough silver summary about completely unrelated topic " * 3
    assert silver_confidence("payment transfer EMEA", _silver_item(content)) == "weak"


def test_silver_confidence_ok_when_overlap():
    content = "This is a long enough silver summary about payment transfer in EMEA region " * 3
    assert silver_confidence("payment transfer EMEA", _silver_item(content)) == "ok"


def test_silver_confidence_ok_when_no_query():
    """Without a query we can only detect missing/low; long content is 'ok'."""
    content = "Some perfectly valid summary that is long enough to be informative. " * 3
    assert silver_confidence(None, _silver_item(content)) == "ok"


def test_silver_item_for_target_picks_silver_only():
    items = [
        ContextItem(path="A", layer="gold", content="gold blurb"),
        ContextItem(path="A", layer="silver", content="silver A"),
        ContextItem(path="B", layer="silver", content="silver B"),
    ]
    item = silver_item_for_target(items, "A")
    assert item is not None
    assert item.content == "silver A"


def test_silver_item_for_target_returns_none_when_missing():
    items = [ContextItem(path="A", layer="gold", content="g")]
    assert silver_item_for_target(items, "A") is None


# ── provider detection ─────────────────────────────────────────────────


class _StubHttpProvider(EmbeddingProvider):
    """Stand-in for `HttpEmbeddingProvider`; not the Mock, so semantic=True.

    Provider is deterministic: a `mapping` of text → vector is consulted first,
    otherwise a fixed default vector is returned. Lets tests stage cosine values
    without spinning up Ollama.
    """

    model_name = "stub-v1"
    embed_dimension = 4

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self.mapping = mapping or {}
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(self.mapping.get(text, [0.0, 0.0, 0.0, 1.0]))


def test_is_semantic_provider_false_for_mock():
    assert is_semantic_provider(MockEmbeddingProvider()) is False


def test_is_semantic_provider_false_for_none():
    assert is_semantic_provider(None) is False


def test_is_semantic_provider_true_for_non_mock():
    assert is_semantic_provider(_StubHttpProvider()) is True


def test_is_semantic_provider_reads_is_semantic_attribute():
    """v1.9: providers can declare `is_semantic = False` without being Mock."""

    class _CustomBagOfWords:
        model_name = "custom-bag"
        embed_dimension = 4
        is_semantic = False
        def embed(self, text):
            return [1.0, 0.0, 0.0, 0.0]

    class _CustomSemantic:
        model_name = "custom-semantic"
        embed_dimension = 4
        is_semantic = True
        def embed(self, text):
            return [0.0, 1.0, 0.0, 0.0]

    assert is_semantic_provider(_CustomBagOfWords()) is False
    assert is_semantic_provider(_CustomSemantic()) is True


def test_is_semantic_provider_legacy_fallback_without_attribute():
    """Pre-v1.9 providers without `is_semantic` are treated as semantic if they
    are not a MockEmbeddingProvider — preserves prior behaviour."""

    class _LegacyProvider:
        model_name = "legacy"
        embed_dimension = 4
        def embed(self, text):
            return [0.0, 0.0, 1.0, 0.0]

    assert is_semantic_provider(_LegacyProvider()) is True


# ── semantic_silver_upgrade ────────────────────────────────────────────


@pytest.mark.parametrize("lexical", ["ok", "missing", "low"])
def test_semantic_silver_upgrade_noop_when_not_weak(lexical):
    assert semantic_silver_upgrade(lexical, [1.0, 0.0], [1.0, 0.0]) == lexical


def test_semantic_silver_upgrade_promotes_weak_when_cosine_high():
    assert semantic_silver_upgrade("weak", [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == "ok"


def test_semantic_silver_upgrade_keeps_weak_when_cosine_low():
    assert semantic_silver_upgrade("weak", [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == "weak"


def test_semantic_silver_upgrade_respects_threshold():
    # Identical unit vectors → cosine = 1.0; threshold above that blocks the upgrade.
    assert (
        semantic_silver_upgrade("weak", [1.0, 0.0], [1.0, 0.0], threshold=1.01) == "weak"
    )


def test_semantic_silver_upgrade_keeps_weak_when_no_vectors():
    assert semantic_silver_upgrade("weak", None, [1.0, 0.0]) == "weak"
    assert semantic_silver_upgrade("weak", [1.0, 0.0], None) == "weak"
    assert semantic_silver_upgrade("weak", [], []) == "weak"


# ── hint builders ──────────────────────────────────────────────────────


class _StubCandidate:
    def __init__(self, path: str):
        self.path = path


def test_route_next_hint_uses_search_semantic_in_both_modes():
    cand = [_StubCandidate("PROJECTS/foo")]
    hint_mock = route_next_hint(cand, semantic=False)
    hint_real = route_next_hint(cand, semantic=True)
    assert hint_mock is not None and "search_semantic" in hint_mock
    assert hint_real is not None and "search_semantic" in hint_real
    assert "no embedding endpoint configured" in hint_mock
    assert "no embedding endpoint configured" not in hint_real


def test_route_next_hint_when_no_candidates():
    hint = route_next_hint([], semantic=True)
    assert hint is not None
    assert "search_semantic" in hint


def test_read_context_next_hint_silent_when_ok():
    assert read_context_next_hint("A", "ok", semantic=False) is None
    assert read_context_next_hint("A", "ok", semantic=True) is None


@pytest.mark.parametrize("level", ["missing", "low", "weak"])
def test_read_context_next_hint_present_when_not_ok(level):
    hint = read_context_next_hint("PROJECTS/foo", level, semantic=True)
    assert hint is not None
    assert "search_semantic" in hint
    assert "PROJECTS/foo" in hint
    assert level in hint


# ── layer bias + suggested paths ───────────────────────────────────────


def _sr(path: str, layer: str, score: float, content_type: str | None = None) -> SearchResult:
    return SearchResult(
        path=path,
        source="chunk",
        score=score,
        snippet="...",
        layer=layer,
        content_type=content_type,
    )


def test_apply_layer_bias_prefers_bronze_over_silver_over_gold():
    raw = [
        _sr("A", "gold", 0.95, "summary"),
        _sr("B", "silver", 0.40, "summary"),
        _sr("C", "bronze", 0.30, "fact"),
    ]
    biased = apply_layer_bias(raw)
    assert [r.layer for r in biased] == ["bronze", "silver", "gold"]


def test_apply_layer_bias_orders_bronze_by_content_type_then_score():
    raw = [
        _sr("A", "bronze", 0.20, "decision"),
        _sr("B", "bronze", 0.50, "fact"),
        _sr("C", "bronze", 0.10, "reference"),
        _sr("D", "bronze", 0.90, "note"),
    ]
    biased = apply_layer_bias(raw)
    assert [r.path for r in biased] == ["C", "B", "A", "D"]


def test_apply_layer_bias_uses_score_within_same_bucket():
    raw = [
        _sr("low", "bronze", 0.10, "fact"),
        _sr("high", "bronze", 0.80, "fact"),
    ]
    assert [r.path for r in apply_layer_bias(raw)] == ["high", "low"]


def test_suggested_paths_dedupes_and_respects_order():
    biased = [
        _sr("A", "bronze", 0.9, "fact"),
        _sr("A", "bronze", 0.5, "decision"),
        _sr("B", "silver", 0.4),
        _sr("C", "gold", 0.2),
    ]
    assert suggested_paths(biased, limit=5) == ["A", "B", "C"]


def test_suggested_paths_respects_limit():
    biased = [_sr(f"N{i}", "bronze", 0.1, "fact") for i in range(10)]
    assert suggested_paths(biased, limit=3) == ["N0", "N1", "N2"]


# ── MCP integration (Mock provider — semantic_endpoint=False) ──────────


def _mcp(tmp_path):
    store = JsonStore(tmp_path)
    return VerticalBrainMCP(store), store


def _mcp_semantic(tmp_path):
    """Same store, but provider that passes the is_semantic_provider check."""
    store = JsonStore(tmp_path)
    return VerticalBrainMCP(store, embedding_provider=_StubHttpProvider()), store


def _call(mcp, name, args):
    return mcp.handle({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
    })


def _text(response: dict) -> str:
    return response["result"]["content"][0]["text"]


def test_read_context_reports_missing_silver_and_hint(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="bronze fact only", layer="bronze"))

    resp = _call(mcp, "read_context", {"path": "WORK/A", "query": "anything"})
    data = json.loads(_text(resp))

    assert data["silver_confidence"] == "missing"
    assert "next_hint" in data
    assert "search_semantic" in data["next_hint"]
    assert "WORK/A" in data["next_hint"]
    assert "no embedding endpoint configured" in data["next_hint"]


def test_read_context_ok_silver_has_no_hint(tmp_path):
    mcp, store = _mcp(tmp_path)
    long_silver = "A thorough summary about payment transfer in EMEA region across countries. " * 4
    store.save_chunk(Chunk(node_path="WORK/B", content=long_silver, layer="silver"))

    resp = _call(mcp, "read_context", {"path": "WORK/B", "query": "payment transfer"})
    data = json.loads(_text(resp))

    assert data["silver_confidence"] == "ok"
    assert "next_hint" not in data


def test_read_context_weak_silver_with_semantic_endpoint(tmp_path):
    mcp, store = _mcp_semantic(tmp_path)
    long_off_topic = "A perfectly fine summary about unrelated topic that is long enough. " * 4
    store.save_chunk(Chunk(node_path="WORK/C", content=long_off_topic, layer="silver"))

    resp = _call(mcp, "read_context", {"path": "WORK/C", "query": "payment transfer EMEA"})
    data = json.loads(_text(resp))

    assert data["silver_confidence"] == "weak"
    assert "next_hint" in data
    assert "no embedding endpoint configured" not in data["next_hint"]


def test_search_semantic_applies_layer_bias(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/D", content="Delta Lake gold tag", layer="gold"))
    store.save_chunk(
        Chunk(node_path="WORK/D", content="Delta Lake reference doc", layer="bronze", content_type="reference"),
    )

    resp = _call(mcp, "search_semantic", {"query": "Delta Lake"})
    data = json.loads(_text(resp))

    layers = [r["layer"] for r in data["results"]]
    assert layers.index("bronze") < layers.index("gold")
    assert "WORK/D" in data["suggested_paths"]


def test_route_response_carries_semantic_flag_and_hint(tmp_path):
    mcp, store = _mcp_semantic(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/E", content="Delta Lake summary", layer="gold"))

    resp = _call(mcp, "route", {"text": "Delta Lake"})
    data = json.loads(_text(resp))

    assert data["semantic_endpoint"] is True
    assert "next_hint" in data
    assert "no embedding endpoint configured" not in data["next_hint"]


# ── v1.6 semantic silver upgrade — MCP integration ─────────────────────


def _save_silver_with_cached_vector(
    store, path: str, content: str, model_name: str, vector: list[float]
):
    """Create a Silver chunk and seed its embedding cache under `model_name`."""
    chunk = Chunk(node_path=path, content=content, layer="silver")
    store.save_chunk(chunk)
    set_vector = getattr(store, "set_vector", None)
    assert callable(set_vector), "store must support set_vector for this test"
    set_vector(chunk.content_hash, model_name, vector)
    return chunk


def test_read_context_semantic_upgrade_promotes_weak_to_ok(tmp_path):
    """Lexical 'weak' (no term overlap) + high cosine via real provider → 'ok'."""
    query = "payment transfer EMEA"
    silver_text = (
        "An extensive note about wire instructions across European countries " * 4
    )
    same_vec = [1.0, 0.0, 0.0, 0.0]
    provider = _StubHttpProvider(mapping={query: same_vec})

    from vertical_brain.storage.json_store import JsonStore
    store = JsonStore(tmp_path)
    _save_silver_with_cached_vector(
        store, "WORK/F", silver_text, provider.model_name, same_vec
    )
    mcp = VerticalBrainMCP(store, embedding_provider=provider)

    resp = _call(mcp, "read_context", {"path": "WORK/F", "query": query})
    data = json.loads(_text(resp))

    assert data["silver_confidence"] == "ok"
    assert "next_hint" not in data
    # The upgrade path must have actually consulted the provider.
    assert provider.calls == [query]


def test_read_context_semantic_upgrade_keeps_weak_when_cosine_low(tmp_path):
    """Lexical 'weak' + low cosine → stays 'weak' (no false promotion)."""
    query = "payment transfer EMEA"
    silver_text = (
        "An extensive note about completely unrelated topic spanning several lines " * 4
    )
    provider = _StubHttpProvider(
        mapping={query: [1.0, 0.0, 0.0, 0.0]},
    )

    from vertical_brain.storage.json_store import JsonStore
    store = JsonStore(tmp_path)
    # Silver cached vector is orthogonal to query vector → cosine = 0.
    _save_silver_with_cached_vector(
        store, "WORK/G", silver_text, provider.model_name, [0.0, 1.0, 0.0, 0.0]
    )
    mcp = VerticalBrainMCP(store, embedding_provider=provider)

    resp = _call(mcp, "read_context", {"path": "WORK/G", "query": query})
    data = json.loads(_text(resp))

    assert data["silver_confidence"] == "weak"
    assert "next_hint" in data
    assert "search_semantic" in data["next_hint"]


def test_read_context_skips_semantic_upgrade_on_mock_provider(tmp_path):
    """Mock provider must NOT trigger the upgrade path (saves an Ollama RTT)."""
    mcp, store = _mcp(tmp_path)
    silver_text = (
        "An extensive note about unrelated topic that is long enough to look fine " * 4
    )
    store.save_chunk(Chunk(node_path="WORK/H", content=silver_text, layer="silver"))

    resp = _call(mcp, "read_context", {"path": "WORK/H", "query": "payment transfer EMEA"})
    data = json.loads(_text(resp))

    # Lexical still says 'weak' (no overlap); Mock provider must not upgrade it.
    assert data["silver_confidence"] == "weak"
    assert "no embedding endpoint configured" in data["next_hint"]


def test_read_context_semantic_upgrade_survives_provider_exception(tmp_path):
    """If provider.embed() raises, fall back to the lexical verdict instead of crashing."""

    class _BrokenProvider(_StubHttpProvider):
        def embed(self, text: str) -> list[float]:
            raise RuntimeError("Ollama is down")

    provider = _BrokenProvider()
    from vertical_brain.storage.json_store import JsonStore
    store = JsonStore(tmp_path)
    _save_silver_with_cached_vector(
        store, "WORK/I",
        "An extensive note about unrelated topic that is long enough to look fine " * 4,
        provider.model_name, [1.0, 0.0, 0.0, 0.0]
    )
    mcp = VerticalBrainMCP(store, embedding_provider=provider)

    resp = _call(mcp, "read_context", {"path": "WORK/I", "query": "payment transfer EMEA"})
    data = json.loads(_text(resp))

    assert data["silver_confidence"] == "weak"  # not crashed, not upgraded
