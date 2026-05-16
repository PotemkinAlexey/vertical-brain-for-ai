from __future__ import annotations

import json
from typing import Optional

from vertical_brain.core.models import Chunk
from vertical_brain.mcp.server import VerticalBrainMCP
from vertical_brain.storage.json_store import JsonStore


def _mcp(tmp_path):
    store = JsonStore(tmp_path)
    return VerticalBrainMCP(store), store


def _call(mcp: VerticalBrainMCP, name: str, args: Optional[dict] = None) -> dict:
    return mcp.handle({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args or {}},
    })


def _text(response: dict) -> str:
    return response["result"]["content"][0]["text"]


def test_initialize_returns_protocol_version(tmp_path):
    mcp, _ = _mcp(tmp_path)
    resp = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert resp["result"]["serverInfo"]["name"] == "vertical-brain"


def test_initialized_notification_returns_none(tmp_path):
    mcp, _ = _mcp(tmp_path)
    resp = mcp.handle({"jsonrpc": "2.0", "method": "initialized"})
    assert resp is None


def test_tools_list_contains_expected_tools(tmp_path):
    mcp, _ = _mcp(tmp_path)
    resp = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = {t["name"] for t in resp["result"]["tools"]}
    assert names == {
        "session_start", "namespace_map", "search", "search_semantic",
        "context_search", "route", "append_chunk", "append_gold_aspect",
    }


def test_unknown_method_returns_error(tmp_path):
    mcp, _ = _mcp(tmp_path)
    resp = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "unknown/method"})
    assert "error" in resp
    assert resp["error"]["code"] == -32601


def test_session_start_returns_orientation_text(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="Delta migration", layer="gold"))

    resp = _call(mcp, "session_start")
    text = _text(resp)

    assert "VERTICAL BRAIN" in text
    assert "WORK/DataArt" in text
    assert "Delta migration" in text


def test_namespace_map_returns_json(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.ensure_node("WORK/DataArt")

    resp = _call(mcp, "namespace_map")
    data = json.loads(_text(resp))

    assert "nodes" in data
    paths = [n["path"] for n in data["nodes"]]
    assert "WORK/DataArt" in paths


def test_append_chunk_writes_and_returns_chunk_id(tmp_path):
    mcp, store = _mcp(tmp_path)

    resp = _call(mcp, "append_chunk", {"path": "WORK/DataArt", "content": "Delta Lake fact", "layer": "silver"})
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    assert result["chunk_id"] is not None
    chunks = store.get_chunks_by_path("WORK/DataArt")
    assert len(chunks) == 1
    assert chunks[0].content == "Delta Lake fact"


def test_append_gold_aspect_writes_gold_chunk(tmp_path):
    mcp, store = _mcp(tmp_path)

    resp = _call(mcp, "append_gold_aspect", {"path": "WORK/DataArt", "aspect": "Delta migration"})
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    gold = [c for c in store.get_chunks_by_path("WORK/DataArt") if c.layer == "gold"]
    assert len(gold) == 1
    assert gold[0].content == "Delta migration"


def test_search_returns_ranked_results(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="Delta Lake streaming ingestion"))

    resp = _call(mcp, "search", {"query": "Delta Lake"})
    results = json.loads(_text(resp))

    assert len(results) >= 1
    assert results[0]["path"] == "WORK/DataArt"
    assert "Delta Lake" in results[0]["snippet"]


def test_search_semantic_returns_similarity_scores(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="Delta Lake streaming ingestion"))

    resp = _call(mcp, "search_semantic", {"query": "Delta streaming"})
    results = json.loads(_text(resp))

    assert len(results) >= 1
    assert 0 < results[0]["score"] <= 1.0


def test_context_search_returns_locked_context_json(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="Delta Lake ingestion fact"))

    resp = _call(mcp, "context_search", {"query": "Delta"})
    data = json.loads(_text(resp))

    assert "locked_contexts" in data
    assert "candidate_handles" in data


def test_route_returns_namespace_candidates(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="Databricks Delta Lake streaming", layer="gold"))

    resp = _call(mcp, "route", {"text": "Delta Lake autoloader"})
    candidates = json.loads(_text(resp))

    assert len(candidates) >= 1
    assert candidates[0]["path"] == "WORK/DataArt"
    assert "score" in candidates[0]


def test_unknown_tool_returns_error(tmp_path):
    mcp, _ = _mcp(tmp_path)
    resp = _call(mcp, "nonexistent_tool")
    assert "error" in resp
    assert resp["error"]["code"] == -32602


def test_response_includes_request_id(tmp_path):
    mcp, _ = _mcp(tmp_path)
    resp = mcp.handle({"jsonrpc": "2.0", "id": 42, "method": "tools/list"})
    assert resp["id"] == 42
