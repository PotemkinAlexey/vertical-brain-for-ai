from __future__ import annotations

import json
from typing import Optional

from vertical_brain.core.models import Chunk
from vertical_brain.mcp.server import VerticalBrainMCP
from vertical_brain.storage.json_store import JsonStore
from vertical_brain.storage.sqlite_store import SQLiteStore


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
        "session_start", "namespace_map", "list_chunks", "read_context",
        "search", "search_semantic", "context_search", "context_search_semantic",
        "route", "append_chunk", "append_gold_aspect", "create_link",
        "mark_stale", "batch_append", "session_end", "optimize",
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
    from vertical_brain.core.gold import parse_gold_content
    assert parse_gold_content(gold[0].content) == ["Delta migration"]


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


def test_list_chunks_returns_full_content(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="Delta Lake fact", layer="silver"))
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="stale chunk", status="stale"))

    resp = _call(mcp, "list_chunks", {"path": "WORK/DataArt"})
    chunks = json.loads(_text(resp))

    assert len(chunks) == 1
    assert chunks[0]["content"] == "Delta Lake fact"
    assert chunks[0]["layer"] == "silver"


def test_list_chunks_include_stale(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="active"))
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="stale", status="stale"))

    resp = _call(mcp, "list_chunks", {"path": "WORK/DataArt", "include_stale": True})
    chunks = json.loads(_text(resp))

    assert len(chunks) == 2


def test_list_chunks_filter_by_layer(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="bronze fact", layer="bronze"))
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="gold label", layer="gold"))

    resp = _call(mcp, "list_chunks", {"path": "WORK/DataArt", "layer": "gold"})
    chunks = json.loads(_text(resp))

    assert len(chunks) == 1
    assert chunks[0]["layer"] == "gold"


def test_read_context_returns_chunks_and_links(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="Delta Lake fact"))

    resp = _call(mcp, "read_context", {"path": "WORK/DataArt"})
    data = json.loads(_text(resp))

    assert "items" in data
    assert any(item["content"] == "Delta Lake fact" for item in data["items"])


def test_create_link_connects_two_namespaces(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.ensure_node("WORK/DataArt/Databricks")
    store.ensure_node("WORK/DataArt/FXDB")

    resp = _call(mcp, "create_link", {
        "source_path": "WORK/DataArt/Databricks",
        "target_path": "WORK/DataArt/FXDB",
        "link_type": "peer",
        "reason": "Related data pipelines.",
    })
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    assert len(result["link_ids"]) == 1
    assert store.get_peer_paths("WORK/DataArt/Databricks") == ["WORK/DataArt/FXDB"]


def test_mark_stale_by_chunk_ids(tmp_path):
    mcp, store = _mcp(tmp_path)
    chunk = store.save_chunk(Chunk(node_path="WORK/DataArt", content="old fact"))

    resp = _call(mcp, "mark_stale", {"path": "WORK/DataArt", "chunk_ids": [chunk.id]})
    result = json.loads(_text(resp))

    assert result["marked"] == 1
    assert store.get_chunks_by_path("WORK/DataArt")[0].status == "stale"


def test_mark_stale_all_active_non_gold_when_no_ids(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="fact one"))
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="fact two"))
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="gold label", layer="gold"))

    resp = _call(mcp, "mark_stale", {"path": "WORK/DataArt"})
    result = json.loads(_text(resp))

    assert result["marked"] == 2
    chunks = store.get_chunks_by_path("WORK/DataArt")
    gold = [c for c in chunks if c.layer == "gold"]
    non_gold = [c for c in chunks if c.layer != "gold"]
    assert gold[0].status == "active"
    assert all(c.status == "stale" for c in non_gold)


def test_append_gold_aspect_reports_overflow_path(tmp_path):
    from vertical_brain.core.gold import MAX_GOLD_ASPECTS, GoldAspect, serialize_gold_aspects
    mcp, store = _mcp(tmp_path)
    full_content = serialize_gold_aspects([GoldAspect(text=f"a{i}") for i in range(MAX_GOLD_ASPECTS)])
    store.save_chunk(Chunk(node_path="WORK/DataArt", content=full_content, layer="gold"))

    resp = _call(mcp, "append_gold_aspect", {"path": "WORK/DataArt", "aspect": "overflow aspect"})
    result = json.loads(_text(resp))

    assert result["overflow_path"] == "WORK/DataArt_2"


def test_batch_append_writes_all_chunks(tmp_path):
    mcp, store = _mcp(tmp_path)

    resp = _call(mcp, "batch_append", {"chunks": [
        {"path": "WORK/A", "content": "fact A", "layer": "silver"},
        {"path": "WORK/B", "content": "fact B", "layer": "bronze"},
    ]})
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    assert len(result["chunk_ids"]) == 2
    assert len(store.get_chunks_by_path("WORK/A")) == 1
    assert len(store.get_chunks_by_path("WORK/B")) == 1


def test_session_end_writes_silver_note(tmp_path):
    mcp, store = _mcp(tmp_path)

    resp = _call(mcp, "session_end", {
        "path": "WORK/DataArt",
        "summary": "Discussed Delta Lake schema evolution.",
    })
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    chunks = store.get_chunks_by_path("WORK/DataArt")
    assert any(c.layer == "silver" and c.content_type == "note" for c in chunks)


def test_session_end_with_gold_aspect(tmp_path):
    mcp, store = _mcp(tmp_path)

    resp = _call(mcp, "session_end", {
        "path": "WORK/DataArt",
        "summary": "Session summary.",
        "gold_aspect": "schema evolution",
    })
    result = json.loads(_text(resp))

    assert result["gold_status"] == "applied"
    gold = [c for c in store.get_chunks_by_path("WORK/DataArt") if c.layer == "gold"]
    from vertical_brain.core.gold import parse_gold_content
    assert parse_gold_content(gold[0].content) == ["schema evolution"]


def test_context_search_semantic_returns_locked_contexts(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="Delta Lake streaming ingestion pipeline"))

    resp = _call(mcp, "context_search_semantic", {"query": "Delta streaming"})
    data = json.loads(_text(resp))

    assert "locked_contexts" in data
    assert "candidate_handles" in data


def test_append_chunk_respects_source_and_confidence(tmp_path):
    mcp, store = _mcp(tmp_path)

    _call(mcp, "append_chunk", {
        "path": "WORK/DataArt",
        "content": "Verified Delta fact.",
        "source": "user",
        "confidence": 0.95,
    })

    chunk = store.get_chunks_by_path("WORK/DataArt")[0]
    assert chunk.source == "user"
    assert chunk.confidence == 0.95


def test_mark_stale_passes_reason_to_audit(tmp_path):
    store = SQLiteStore(tmp_path)
    mcp = VerticalBrainMCP(store)
    chunk = store.save_chunk(Chunk(node_path="WORK/DataArt", content="outdated fact"))

    _call(mcp, "mark_stale", {
        "path": "WORK/DataArt",
        "chunk_ids": [chunk.id],
        "reason": "superseded by new analysis",
    })

    audit = store.list_audit()
    assert len(audit) == 1
    assert audit[0]["reasoning_summary"] == "superseded by new analysis"


def test_batch_append_uses_atomic_transaction_sqlite(tmp_path):
    store = SQLiteStore(tmp_path)
    mcp = VerticalBrainMCP(store)

    resp = _call(mcp, "batch_append", {"chunks": [
        {"path": "WORK/A", "content": "fact A", "layer": "silver"},
        {"path": "WORK/B", "content": "fact B"},
        {"path": "WORK/C", "content": "fact C"},
    ]})
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    assert len(result["chunk_ids"]) == 3
    assert len(store.get_chunks_by_path("WORK/A")) == 1
    assert len(store.get_chunks_by_path("WORK/B")) == 1
    assert len(store.get_chunks_by_path("WORK/C")) == 1


def test_batch_append_writes_audit_for_each_operation(tmp_path):
    store = SQLiteStore(tmp_path)
    mcp = VerticalBrainMCP(store)

    _call(mcp, "batch_append", {
        "reasoning_summary": "batch rationale",
        "chunks": [
            {"path": "WORK/A", "content": "one"},
            {"path": "WORK/B", "content": "two"},
        ],
    })

    audit = store.list_audit()
    assert len(audit) == 2
    assert all(r["reasoning_summary"] == "batch rationale" for r in audit)
    assert all(r["status"] == "applied" for r in audit)


def test_batch_append_is_all_or_nothing_on_validation_failure(tmp_path):
    """A batch with one invalid item (empty content) must write nothing and create no audit."""
    store = SQLiteStore(tmp_path)
    mcp = VerticalBrainMCP(store)

    resp = _call(mcp, "batch_append", {"chunks": [
        {"path": "WORK/A", "content": "valid fact"},
        {"path": "WORK/B", "content": ""},  # invalid: empty content
    ]})

    # MCP returns an error, not a success result.
    assert "error" in resp
    # No chunks written anywhere.
    assert store.get_chunks_by_path("WORK/A") == []
    assert store.get_chunks_by_path("WORK/B") == []
    # No audit records created.
    assert store.list_audit() == []
