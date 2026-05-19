from __future__ import annotations

import json
from io import BytesIO
from typing import Optional

from vertical_brain.core.models import Chunk
from vertical_brain.mcp.server import MessageParseError, VerticalBrainMCP, _read_message, _write_message
from vertical_brain.storage.json_store import JsonStore
from vertical_brain.storage.sqlite_store import SQLiteStore


def _mcp(tmp_path):
    store = JsonStore(tmp_path)
    return VerticalBrainMCP(store), store


def _mcp_from_store(store):
    return VerticalBrainMCP(store)


def _call(mcp: VerticalBrainMCP, name: str, args: Optional[dict] = None) -> dict:
    return mcp.handle({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args or {}},
    })


def _text(response: dict) -> str:
    return response["result"]["content"][0]["text"]


_SKIP_OK = "page footer copyright notice only, no operational data"


def _parse_session_key(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("session_key:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError("session_key not found in ingest response")


def _complete_routing_session(mcp: VerticalBrainMCP, session_key: str) -> None:
    for chunk in mcp._ingest_sessions[session_key]["chunks"]:
        _call(
            mcp,
            "mark_service_chunk",
            {
                "session_key": session_key,
                "chunk_id": chunk["id"],
                "status": "skipped",
                "skip_reason": _SKIP_OK,
            },
        )
    _call(mcp, "finish_bronze_extraction", {"session_key": session_key})
    _call(mcp, "complete_ingest", {"session_key": session_key})


def test_initialize_returns_protocol_version(tmp_path):
    mcp, _ = _mcp(tmp_path)
    resp = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert resp["result"]["protocolVersion"] == "2025-03-26"
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
        "mark_stale", "batch_append", "session_end", "update_silver", "optimize",
        "operations", "doctor", "vacuum", "ingest_file", "ingest_url",
        "get_service_chunk", "get_service_chunks", "mark_service_chunk",
        "batch_mark_service_chunks",
        "finish_bronze_extraction", "submit_inventory_probes", "complete_ingest",
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


def test_session_start_slash_root_path_returns_full_tree(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="Delta migration", layer="gold"))
    store.save_chunk(Chunk(node_path="PERSONAL/Blog", content="Personal note"))

    resp = _call(mcp, "session_start", {"root_path": "/", "max_depth": 4})
    text = _text(resp)

    assert "WORK/DataArt" in text
    assert "PERSONAL/Blog" in text
    assert "0 nodes · 0 active chunks" not in text


def test_session_start_returns_agents_md_contract(tmp_path, monkeypatch):
    agents_file = tmp_path / "AGENTS.md"
    agents_file.write_text("# Agent Contract\n\nSearch before Bronze writes.", encoding="utf-8")
    monkeypatch.setenv("VERTICAL_BRAIN_AGENTS_PATH", str(agents_file))
    mcp, _ = _mcp(tmp_path / "store")

    resp = _call(mcp, "session_start")
    text = _text(resp)

    assert text.startswith("AGENTS.md\n\n# Agent Contract")
    assert "Search before Bronze writes." in text
    assert "---\n\nVERTICAL BRAIN" in text


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
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="silver summary", layer="silver"))

    resp = _call(mcp, "append_gold_aspect", {"path": "WORK/DataArt", "aspect": "Delta migration"})
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    gold = [c for c in store.get_chunks_by_path("WORK/DataArt") if c.layer == "gold"]
    assert len(gold) == 1
    from vertical_brain.core.gold import parse_gold_content
    assert parse_gold_content(gold[0].content) == ["Delta migration"]


def test_append_gold_aspect_reports_aspect_too_long(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="silver summary", layer="silver"))

    resp = _call(mcp, "append_gold_aspect", {"path": "WORK/DataArt", "aspect": "x" * 151})
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    assert result["aspects_too_long"] == ["x" * 151]


def test_append_gold_aspect_accepts_multiple_aspects(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="silver summary", layer="silver"))

    resp = _call(mcp, "append_gold_aspect", {
        "path": "WORK/DataArt",
        "aspects": ["routing tag one", "routing tag two", "routing tag three"],
    })
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    assert result["aspects_added"] == 3


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
    data = json.loads(_text(resp))

    results = data["results"]
    assert len(results) >= 1
    assert 0 < results[0]["score"] <= 1.0
    assert isinstance(data["suggested_paths"], list)
    assert data["suggested_paths"][0] == "WORK/DataArt"
    assert data["semantic_endpoint"] is False  # default Mock provider


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
    data = json.loads(_text(resp))

    candidates = data["candidates"]
    assert len(candidates) >= 1
    assert candidates[0]["path"] == "WORK/DataArt"
    assert "score" in candidates[0]
    assert data["semantic_endpoint"] is False
    assert "next_hint" in data
    assert "WORK/DataArt" in data["next_hint"]


def test_unknown_tool_returns_error(tmp_path):
    mcp, _ = _mcp(tmp_path)
    resp = _call(mcp, "nonexistent_tool")
    assert "error" in resp
    assert resp["error"]["code"] == -32602
    assert "Unknown tool" in resp["error"]["message"]


def test_tool_call_rejects_missing_required_argument(tmp_path):
    mcp, _ = _mcp(tmp_path)

    resp = _call(mcp, "list_chunks", {})

    assert "error" in resp
    assert resp["error"]["code"] == -32602
    assert "$.path: is required" in resp["error"]["message"]


def test_tool_call_rejects_bad_enum_argument(tmp_path):
    mcp, _ = _mcp(tmp_path)

    resp = _call(mcp, "read_context", {"path": "WORK/A", "link_expansion": "everything"})

    assert "error" in resp
    assert resp["error"]["code"] == -32602
    assert "must be one of: handles_only, expanded, none" in resp["error"]["message"]


def test_tool_call_rejects_non_object_arguments(tmp_path):
    mcp, _ = _mcp(tmp_path)

    resp = mcp.handle({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "search", "arguments": "Delta"},
    })

    assert "error" in resp
    assert resp["error"]["code"] == -32602
    assert "arguments must be an object" in resp["error"]["message"]


def test_tool_call_rejects_non_object_params(tmp_path):
    mcp, _ = _mcp(tmp_path)

    resp = mcp.handle({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": "search",
    })

    assert "error" in resp
    assert resp["error"]["code"] == -32602
    assert "params must be an object" in resp["error"]["message"]


def test_response_includes_request_id(tmp_path):
    mcp, _ = _mcp(tmp_path)
    resp = mcp.handle({"jsonrpc": "2.0", "id": 42, "method": "tools/list"})
    assert resp["id"] == 42


def test_read_message_parses_newline_delimited_json():
    stream = BytesIO(b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n')

    payload = _read_message(stream)

    assert payload["method"] == "ping"
    assert payload["id"] == 1


def test_read_message_skips_blank_lines():
    stream = BytesIO(b'\n\n{"jsonrpc":"2.0","id":2,"method":"ping"}\n')

    payload = _read_message(stream)

    assert payload["id"] == 2


def test_read_message_rejects_invalid_json():
    stream = BytesIO(b"not-json\n")

    try:
        _read_message(stream)
    except MessageParseError as exc:
        assert "Invalid JSON" in str(exc)
    else:
        raise AssertionError("expected MessageParseError")


def test_read_message_rejects_non_object_body():
    stream = BytesIO(b"[]\n")

    try:
        _read_message(stream)
    except MessageParseError as exc:
        assert "JSON object" in str(exc)
    else:
        raise AssertionError("expected MessageParseError")


def test_write_message_emits_newline_delimited_json():
    stream = BytesIO()

    _write_message(stream, {"jsonrpc": "2.0", "id": 1, "result": {}})

    payload = stream.getvalue()
    assert payload.endswith(b"\n")
    assert b"Content-Length" not in payload
    import json
    parsed = json.loads(payload.decode("utf-8"))
    assert parsed == {"jsonrpc": "2.0", "id": 1, "result": {}}


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
    # Single-chunk items must carry their chunk_id so callers can feed update_silver.
    fact_item = next(item for item in data["items"] if item["content"] == "Delta Lake fact")
    assert fact_item["chunk_id"] == store.get_chunks_by_path("WORK/DataArt")[0].id


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
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="silver summary", layer="silver"))
    full_content = serialize_gold_aspects([GoldAspect(text=f"a{i}") for i in range(MAX_GOLD_ASPECTS)])
    store.save_chunk(Chunk(node_path="WORK/DataArt", content=full_content, layer="gold"))

    resp = _call(mcp, "append_gold_aspect", {"path": "WORK/DataArt", "aspect": "overflow aspect"})
    result = json.loads(_text(resp))

    assert result["overflow_paths"] == ["WORK/DataArt_2"]


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


def test_session_end_summary_only_writes_bronze(tmp_path):
    """Without notes, summary is stored as Bronze — Silver promotion is left to the optimizer."""
    mcp, store = _mcp(tmp_path)

    resp = _call(mcp, "session_end", {
        "path": "WORK/DataArt",
        "summary": "Discussed Delta Lake schema evolution.",
    })
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    chunks = store.get_chunks_by_path("WORK/DataArt")
    assert any(c.layer == "bronze" and c.content_type == "note" for c in chunks)
    assert not any(c.layer == "silver" for c in chunks)


def test_session_end_with_notes_writes_bronze_then_silver(tmp_path):
    """With notes + summary: Bronze = raw notes, Silver = refined summary."""
    mcp, store = _mcp(tmp_path)

    resp = _call(mcp, "session_end", {
        "path": "WORK/DataArt",
        "notes": "Raw session facts: added merge schema, tested on 3 tables.",
        "summary": "Confirmed mergeSchema=true pattern for schema evolution.",
    })
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    chunks = store.get_chunks_by_path("WORK/DataArt")
    assert any(c.layer == "bronze" for c in chunks)
    assert any(c.layer == "silver" for c in chunks)
    bronze = next(c for c in chunks if c.layer == "bronze")
    silver = next(c for c in chunks if c.layer == "silver")
    assert "Raw session facts" in bronze.content
    assert "mergeSchema" in silver.content


def test_session_end_updates_existing_silver(tmp_path):
    """A second session_end on a namespace must rewrite the Silver, not fail
    on the append_chunk(layer=silver) guard."""
    mcp, store = _mcp(tmp_path)

    _call(mcp, "session_end", {
        "path": "WORK/DataArt",
        "notes": "First session raw notes.",
        "summary": "First Silver summary.",
    })
    resp = _call(mcp, "session_end", {
        "path": "WORK/DataArt",
        "notes": "Second session raw notes.",
        "summary": "Second Silver summary.",
    })
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    active_silver = [
        c for c in store.get_chunks_by_path("WORK/DataArt")
        if c.layer == "silver" and c.status == "active"
    ]
    assert len(active_silver) == 1
    assert active_silver[0].content == "Second Silver summary."


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


def test_mark_stale_recursive_marks_children(tmp_path):
    """mark_stale with recursive=true marks chunks in descendant namespaces."""
    store = SQLiteStore(tmp_path)
    mcp = VerticalBrainMCP(store)
    store.save_chunk(Chunk(node_path="SOURCES/BofA", content="parent fact"))
    store.save_chunk(Chunk(node_path="SOURCES/BofA/north-america", content="child fact"))
    store.save_chunk(Chunk(node_path="SOURCES/BofA/emea", content="emea fact"))

    resp = _call(mcp, "mark_stale", {"path": "SOURCES/BofA", "recursive": True, "reason": "redo"})
    result = json.loads(_text(resp))

    assert result["marked"] == 3
    all_chunks = (
        store.get_chunks_by_path("SOURCES/BofA") +
        store.get_chunks_by_path("SOURCES/BofA/north-america") +
        store.get_chunks_by_path("SOURCES/BofA/emea")
    )
    assert all(c.status == "stale" for c in all_chunks)


def test_mark_stale_recursive_skips_immutable(tmp_path):
    """mark_stale with recursive=true silently skips immutable chunks."""
    store = SQLiteStore(tmp_path)
    mcp = VerticalBrainMCP(store)
    store.save_chunk(Chunk(node_path="SOURCES/BofA", content="mutable fact", immutable=False))
    store.save_chunk(Chunk(node_path="SOURCES/BofA/sub", content="immutable artifact", immutable=True))

    resp = _call(mcp, "mark_stale", {"path": "SOURCES/BofA", "recursive": True})
    result = json.loads(_text(resp))

    assert result["marked"] == 1
    immutable_chunk = store.get_chunks_by_path("SOURCES/BofA/sub")[0]
    assert immutable_chunk.status == "active"  # untouched


def test_mark_stale_include_immutable_wipes_everything(tmp_path):
    """include_immutable=true allows full wipe of a namespace subtree for re-ingestion."""
    store = SQLiteStore(tmp_path)
    mcp = VerticalBrainMCP(store)
    store.save_chunk(Chunk(node_path="SOURCES/BofA", content="mutable fact", immutable=False))
    store.save_chunk(Chunk(node_path="SOURCES/BofA/sub", content="immutable artifact", immutable=True))

    resp = _call(mcp, "mark_stale", {
        "path": "SOURCES/BofA",
        "recursive": True,
        "include_immutable": True,
        "reason": "wiping for re-ingestion",
    })
    result = json.loads(_text(resp))

    assert result["marked"] == 2
    all_chunks = (
        store.get_chunks_by_path("SOURCES/BofA") +
        store.get_chunks_by_path("SOURCES/BofA/sub")
    )
    assert all(c.status == "stale" for c in all_chunks)


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


# ── operations tool ───────────────────────────────────────────────────────────

def test_operations_tool_applies_batch_and_returns_status(tmp_path):
    """operations tool executes a StorageOperationBatch and returns status=applied."""
    store = SQLiteStore(tmp_path)
    mcp = VerticalBrainMCP(store)

    payload = {
        "operations": [
            {"operation": "append_chunk", "target_path": "WORK/Q", "chunk": {"content": "ops fact"}},
        ]
    }
    resp = _call(mcp, "operations", {"payload": payload})
    assert "error" not in resp

    data = json.loads(_text(resp))
    assert data["status"] == "applied"
    # Chunk actually landed.
    chunks = store.get_chunks_by_path("WORK/Q")
    assert any(c.content == "ops fact" for c in chunks)


def test_operations_tool_returns_per_operation_results(tmp_path):
    """operations tool response includes a results array with one entry per operation."""
    store = SQLiteStore(tmp_path)
    mcp = VerticalBrainMCP(store)

    payload = {
        "operations": [
            {"operation": "append_chunk", "target_path": "WORK/P", "chunk": {"content": "first"}},
            {"operation": "append_chunk", "target_path": "WORK/P", "chunk": {"content": "second"}},
        ]
    }
    resp = _call(mcp, "operations", {"payload": payload})
    data = json.loads(_text(resp))
    assert len(data["results"]) == 2
    assert all(r["status"] == "applied" for r in data["results"])


def test_operations_tool_rejects_invalid_payload_type(tmp_path):
    """operations tool must reject payload that is not a JSON object."""
    mcp, _ = _mcp(tmp_path)
    resp = _call(mcp, "operations", {"payload": "not-an-object"})
    assert "error" in resp


# ── doctor tool ───────────────────────────────────────────────────────────────

def test_doctor_returns_empty_list_for_clean_store(tmp_path):
    """doctor on a freshly initialised store reports no issues."""
    mcp, _ = _mcp(tmp_path)
    resp = _call(mcp, "doctor", {})
    assert "error" not in resp
    issues = json.loads(_text(resp))
    assert isinstance(issues, list)
    assert issues == []


def test_doctor_detects_duplicate_active_chunks(tmp_path):
    """doctor flags duplicate active chunks sharing the same content at the same node."""
    store = JsonStore(tmp_path)
    mcp = VerticalBrainMCP(store)

    # Save the same content twice — both chunks land as active, triggering the check.
    store.save_chunk(Chunk(node_path="WORK/Dup", content="repeated fact"))
    store.save_chunk(Chunk(node_path="WORK/Dup", content="repeated fact"))

    resp = _call(mcp, "doctor", {})
    assert "error" not in resp
    issues = json.loads(_text(resp))
    assert len(issues) >= 1
    checks = [i["check"] for i in issues]
    assert "duplicate_active_chunk" in checks
    # Every issue has required fields.
    for issue in issues:
        assert "severity" in issue
        assert "check" in issue
        assert "message" in issue


# ── vacuum tool ───────────────────────────────────────────────────────────────

def test_vacuum_tool_defaults_to_dry_run(tmp_path):
    store = SQLiteStore(tmp_path)
    mcp = VerticalBrainMCP(store)
    stale = store.save_chunk(
        Chunk(
            node_path="WORK/Old",
            content="obsolete fact",
            status="stale",
            valid_to="2000-01-01T00:00:00+00:00",
            updated_at="2000-01-01T00:00:00+00:00",
        )
    )

    resp = _call(mcp, "vacuum", {"retention_hours": 0})
    data = json.loads(_text(resp))

    assert data["dry_run"] is True
    assert data["eligible_chunks"] == 1
    assert store.get_chunks_by_path("WORK/Old")[0].id == stale.id


def test_vacuum_tool_can_apply_with_force(tmp_path):
    store = SQLiteStore(tmp_path)
    mcp = VerticalBrainMCP(store)
    store.save_chunk(
        Chunk(
            node_path="WORK/Old",
            content="obsolete fact",
            status="stale",
            valid_to="2000-01-01T00:00:00+00:00",
            updated_at="2000-01-01T00:00:00+00:00",
        )
    )

    resp = _call(mcp, "vacuum", {"retention_hours": 0, "dry_run": False, "force": True})
    data = json.loads(_text(resp))

    assert data["dry_run"] is False
    assert data["deleted_chunks"] == 1
    assert store.get_chunks_by_path("WORK/Old") == []


def test_vacuum_tool_can_include_immutable(tmp_path):
    store = SQLiteStore(tmp_path)
    mcp = VerticalBrainMCP(store)
    store.save_chunk(
        Chunk(
            node_path="WORK/Old",
            content="immutable obsolete fact",
            status="stale",
            immutable=True,
            valid_to="2000-01-01T00:00:00+00:00",
            updated_at="2000-01-01T00:00:00+00:00",
        )
    )

    resp = _call(mcp, "vacuum", {
        "retention_hours": 0,
        "dry_run": False,
        "force": True,
        "include_immutable": True,
    })
    data = json.loads(_text(resp))

    assert data["include_immutable"] is True
    assert data["deleted_chunks"] == 1
    assert store.get_chunks_by_path("WORK/Old") == []


# ── batch_append default reasoning_summary ────────────────────────────────────

def test_batch_append_default_reasoning_summary_appears_in_audit(tmp_path):
    """When no reasoning_summary is provided, audit record uses the MCP default string."""
    store = SQLiteStore(tmp_path)
    mcp = VerticalBrainMCP(store)

    _call(mcp, "batch_append", {"chunks": [{"path": "WORK/R", "content": "audited fact"}]})

    audit = store.list_audit()
    assert len(audit) == 1
    assert audit[0]["reasoning_summary"] == "Batch appended via MCP batch_append."


def test_update_silver_supersedes_old_and_writes_new(tmp_path):
    """Existing Silver — new call supersedes old, writes new."""
    mcp, store = _mcp(tmp_path)
    old_silver = store.save_chunk(Chunk(node_path="PROJECTS/alpha", content="old summary", layer="silver"))

    resp = _call(mcp, "update_silver", {
        "path": "PROJECTS/alpha",
        "new_content": "Updated summary with new fact.",
        "current_silver_id": old_silver.id,
    })
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    assert "chunk_id" in result
    assert result["chunk_id"] is not None
    chunks = store.get_chunks_by_path("PROJECTS/alpha")
    active_silver = [c for c in chunks if c.layer == "silver" and c.status == "active"]
    assert len(active_silver) == 1
    assert active_silver[0].content == "Updated summary with new fact."
    assert next(c for c in chunks if c.id == old_silver.id).status == "superseded"


def test_update_silver_with_source_chunk_ids(tmp_path):
    """source_chunk_ids are recorded in new Silver lineage."""
    mcp, store = _mcp(tmp_path)
    old_silver = store.save_chunk(Chunk(node_path="PROJECTS/alpha", content="old summary", layer="silver"))
    bronze = store.save_chunk(Chunk(node_path="PROJECTS/alpha", content="raw fact", layer="bronze"))

    resp = _call(mcp, "update_silver", {
        "path": "PROJECTS/alpha",
        "new_content": "Refined summary of raw fact.",
        "current_silver_id": old_silver.id,
        "source_chunk_ids": [bronze.id],
    })
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    chunks = store.get_chunks_by_path("PROJECTS/alpha")
    new_silver = next(c for c in chunks if c.id == result["chunk_id"])
    assert bronze.id in new_silver.lineage
    assert old_silver.id in new_silver.lineage


def test_update_silver_occ_rejects_stale_id(tmp_path):
    """OCC: passing a wrong current_silver_id returns an error."""
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="PROJECTS/alpha", content="actual silver", layer="silver"))

    resp = _call(mcp, "update_silver", {
        "path": "PROJECTS/alpha",
        "new_content": "New summary.",
        "current_silver_id": "nonexistent-id",
    })
    assert "error" in resp


def test_update_silver_occ_rejects_null_current_silver_id(tmp_path):
    """Passing null current_silver_id returns a validation error."""
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="PROJECTS/alpha", content="existing silver", layer="silver"))

    resp = _call(mcp, "update_silver", {
        "path": "PROJECTS/alpha",
        "new_content": "New summary.",
        "current_silver_id": None,
    })
    assert "error" in resp


def test_update_silver_patch_applies_single_edit(tmp_path):
    """patch mode edits the current Silver in place — no full resend, no current_silver_id."""
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="PROJECTS/alpha", content="State: alpha. Tests: 10 pass.", layer="silver"))

    resp = _call(mcp, "update_silver", {
        "path": "PROJECTS/alpha",
        "patch": [{"find": "10 pass", "replace": "11 pass"}],
    })
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    active = [c for c in store.get_chunks_by_path("PROJECTS/alpha") if c.layer == "silver" and c.status == "active"]
    assert len(active) == 1
    assert active[0].content == "State: alpha. Tests: 11 pass."


def test_update_silver_patch_rejects_missing_find(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="PROJECTS/alpha", content="alpha summary", layer="silver"))

    resp = _call(mcp, "update_silver", {
        "path": "PROJECTS/alpha",
        "patch": [{"find": "not in the silver", "replace": "x"}],
    })
    assert "error" in resp


def test_update_silver_patch_rejects_ambiguous_find(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="PROJECTS/alpha", content="fact one. fact two.", layer="silver"))

    resp = _call(mcp, "update_silver", {
        "path": "PROJECTS/alpha",
        "patch": [{"find": "fact", "replace": "note"}],
    })
    assert "error" in resp


def test_update_silver_rejects_both_patch_and_new_content(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="PROJECTS/alpha", content="summary", layer="silver"))

    resp = _call(mcp, "update_silver", {
        "path": "PROJECTS/alpha",
        "new_content": "full rewrite",
        "patch": [{"find": "summary", "replace": "x"}],
    })
    assert "error" in resp


def test_update_silver_rejects_neither_patch_nor_new_content(tmp_path):
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="PROJECTS/alpha", content="summary", layer="silver"))

    resp = _call(mcp, "update_silver", {"path": "PROJECTS/alpha"})
    assert "error" in resp


def test_update_silver_reports_silver_too_large(tmp_path):
    """An oversized Silver returns silver_too_large — the namespace-overload signal."""
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="PROJECTS/alpha", content="seed", layer="silver"))

    resp = _call(mcp, "update_silver", {
        "path": "PROJECTS/alpha",
        "new_content": "x" * 2100,
        "current_silver_id": store.get_chunks_by_path("PROJECTS/alpha")[0].id,
    })
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    assert result["silver_too_large"] is True


def test_append_gold_aspect_reports_gold_near_limit(tmp_path):
    """Gold aspect count near the 20 cap returns gold_near_limit."""
    mcp, store = _mcp(tmp_path)
    store.save_chunk(Chunk(node_path="PROJECTS/alpha", content="silver summary", layer="silver"))

    resp = _call(mcp, "append_gold_aspect", {
        "path": "PROJECTS/alpha",
        "aspects": [f"routing tag number {i}" for i in range(16)],
    })
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    assert result["gold_near_limit"] is True


def test_update_silver_audit_record_has_correct_operation_type(tmp_path):
    """MCP update_silver audit record must have operation_type='update_silver', not two records."""
    mcp, store = _mcp(tmp_path)
    old_silver = store.save_chunk(Chunk(node_path="PROJECTS/alpha", content="v1", layer="silver"))

    _call(mcp, "update_silver", {
        "path": "PROJECTS/alpha",
        "new_content": "v2",
        "current_silver_id": old_silver.id,
    })

    audit = store.list_audit()
    silver_ops = [a for a in audit if a.get("operation_type") == "update_silver"]
    assert len(silver_ops) == 1, f"Expected exactly one update_silver audit record, got: {audit}"


def test_update_silver_returns_valid_chunk_id(tmp_path):
    """MCP update_silver returns status=applied and a valid chunk_id."""
    mcp, store = _mcp(tmp_path)
    old_silver = store.save_chunk(Chunk(node_path="PROJECTS/alpha", content="v1", layer="silver"))

    resp = _call(mcp, "update_silver", {
        "path": "PROJECTS/alpha",
        "new_content": "v2",
        "current_silver_id": old_silver.id,
    })
    result = json.loads(_text(resp))

    assert result["status"] == "applied"
    chunk_id = result["chunk_id"]
    assert isinstance(chunk_id, str) and chunk_id
    chunks = store.get_chunks_by_path("PROJECTS/alpha")
    assert any(c.id == chunk_id and c.layer == "silver" and c.status == "active" for c in chunks)


def test_ingest_file_returns_metadata_header(tmp_path):
    """ingest_file creates an ingest session with service chunks and inline protocol."""
    mcp, _ = _mcp(tmp_path)
    file_content = "Field 32A: Value Date, Currency, Amount. Format: 6!n3!a15d"

    resp = _call(mcp, "ingest_file", {
        "content": file_content,
        "file_name": "MT103.txt",
        "authority": "SWIFT",
        "doc_slug": "MT103",
    })
    text = _text(resp)

    assert "MT103.txt" in text
    assert "SOURCES/MT103" in text
    assert "content_sha256" in text
    assert "session_key:" in text
    assert "Service chunks" in text
    assert "IRON RULES" in text
    assert "ingest_mode: answer_complete" in text
    assert "[INVENTORY]" in text


def test_answer_complete_rejects_all_skipped_finish(tmp_path):
    mcp, _ = _mcp(tmp_path)
    resp = _call(mcp, "ingest_file", {
        "content": "Field 32A: Value Date, Currency, Amount.\n\nField 59: Beneficiary.",
        "file_name": "MT103.txt",
    })
    session_key = _parse_session_key(_text(resp))
    for chunk in mcp._ingest_sessions[session_key]["chunks"]:
        _call(
            mcp,
            "mark_service_chunk",
            {
                "session_key": session_key,
                "chunk_id": chunk["id"],
                "status": "skipped",
                "skip_reason": _SKIP_OK,
            },
        )
    result = _call(mcp, "finish_bronze_extraction", {"session_key": session_key})
    assert result.get("error") or result.get("result", {}).get("isError")
    assert "at least one extracted" in json.dumps(result)


def test_mark_service_chunk_rejects_bulk_close_skip_reason(tmp_path):
    mcp, _ = _mcp(tmp_path)
    resp = _call(mcp, "ingest_file", {
        "content": "Field 32A: Value Date, Currency, Amount. Format: 6!n3!a15d",
        "file_name": "MT103.txt",
        "authority": "SWIFT",
        "doc_slug": "MT103",
    })
    session_key = _parse_session_key(_text(resp))
    chunk_id = mcp._ingest_sessions[session_key]["chunks"][0]["id"]

    result = _call(mcp, "mark_service_chunk", {
        "session_key": session_key,
        "chunk_id": chunk_id,
        "status": "skipped",
        "skip_reason": "bulk skip for clean re-ingest verification",
    })

    assert result.get("error") or (result.get("result", {}).get("isError"))
    assert "not a content reason" in json.dumps(result)


def test_ingest_file_defaults_slug_to_filename(tmp_path):
    mcp, _ = _mcp(tmp_path)

    resp = _call(mcp, "ingest_file", {
        "content": "# Spec\nSome content here.",
        "file_name": "my_spec.md",
    })
    text = _text(resp)

    assert "my_spec" in text          # slug derived from filename


def test_ingest_file_rejects_missing_required_args(tmp_path):
    mcp, _ = _mcp(tmp_path)
    resp = _call(mcp, "ingest_file", {"file_name": "doc.txt"})  # missing content
    assert resp.get("error") or (resp.get("result", {}).get("isError"))


def test_batch_mark_service_chunks_marks_all(tmp_path):
    mcp, _ = _mcp(tmp_path)
    resp = _call(mcp, "ingest_file", {
        "content": "Field 32A: Value Date.\n\nField 50K: Ordering Customer.\n\nField 59: Beneficiary.",
        "file_name": "MT103.txt",
        "authority": "SWIFT",
        "doc_slug": "MT103",
    })
    session_key = _parse_session_key(_text(resp))
    chunks = mcp._ingest_sessions[session_key]["chunks"]
    marks = [{"chunk_id": c["id"], "status": "extracted"} for c in chunks]
    result = _call(mcp, "batch_mark_service_chunks", {"session_key": session_key, "marks": marks})
    assert "error" not in result
    data = json.loads(result["result"]["content"][0]["text"])
    assert data["ok"] is True
    assert data["marked"] == len(chunks)
    assert data["remaining_pending"] == 0


def test_batch_mark_service_chunks_rejects_bad_skip(tmp_path):
    mcp, _ = _mcp(tmp_path)
    resp = _call(mcp, "ingest_file", {
        "content": "Some payment field data here.",
        "file_name": "spec.txt",
    })
    session_key = _parse_session_key(_text(resp))
    chunk_id = mcp._ingest_sessions[session_key]["chunks"][0]["id"]
    result = _call(mcp, "batch_mark_service_chunks", {
        "session_key": session_key,
        "marks": [{"chunk_id": chunk_id, "status": "skipped"}],  # missing skip_reason
    })
    assert result.get("error") or result.get("result", {}).get("isError")


def test_batch_mark_service_chunks_rejects_over_limit(tmp_path):
    mcp, _ = _mcp(tmp_path)
    resp = _call(mcp, "ingest_file", {
        "content": "Some content.",
        "file_name": "spec.txt",
    })
    session_key = _parse_session_key(_text(resp))
    marks = [{"chunk_id": f"fake_{i}", "status": "extracted"} for i in range(51)]
    result = _call(mcp, "batch_mark_service_chunks", {"session_key": session_key, "marks": marks})
    assert result.get("error") or result.get("result", {}).get("isError")


def test_ingest_file_hash_is_sha256(tmp_path):
    import hashlib
    mcp, _ = _mcp(tmp_path)
    content = "deterministic content for hashing"
    expected_hash = hashlib.sha256(content.encode()).hexdigest()

    resp = _call(mcp, "ingest_file", {"content": content, "file_name": "doc.txt"})
    text = _text(resp)

    assert expected_hash in text  # full sha256 on content_sha256 line


def test_ingest_file_can_read_source_path(tmp_path):
    import hashlib
    mcp, _ = _mcp(tmp_path)
    source = tmp_path / "spec.txt"
    content = "complete source text\nwith two lines\n"
    source.write_text(content, encoding="utf-8")
    expected_hash = hashlib.sha256(content.encode()).hexdigest()

    resp = _call(mcp, "ingest_file", {"source_path": str(source), "authority": "Internal"})
    text = _text(resp)

    assert "session_key:" in text
    assert expected_hash in text
    assert "SOURCES/spec" in text


def test_ingest_file_rejects_pdf_content_without_source_path(tmp_path):
    mcp, _ = _mcp(tmp_path)

    resp = _call(mcp, "ingest_file", {
        "content": "short summary of a much larger PDF",
        "file_name": "Global-PA-Payments.pdf",
    })

    assert resp.get("error") or (resp.get("result", {}).get("isError"))
    text = json.dumps(resp)
    assert "ingestion must use source_path" in text


def test_ingest_file_rejects_integrity_mismatch(tmp_path):
    mcp, _ = _mcp(tmp_path)

    resp = _call(mcp, "ingest_file", {
        "content": "actual payload",
        "file_name": "doc.txt",
        "expected_size_bytes": 999,
    })

    assert resp.get("error") or (resp.get("result", {}).get("isError"))
    text = json.dumps(resp)
    assert "integrity check failed" in text


def test_ingest_file_duplicate_rejected(tmp_path):
    """Second ingest of same content is rejected unless force=true."""
    mcp = _mcp_from_store(SQLiteStore(root=tmp_path))
    content = "Field 32A: value date, currency, amount."

    # First ingest — create and complete the session
    resp = _call(mcp, "ingest_file", {
        "content": content,
        "file_name": "MT103.txt",
        "mode": "routing",
    })
    session_key = _parse_session_key(_text(resp))
    _complete_routing_session(mcp, session_key)

    # Second ingest — same content, should be rejected
    resp2 = _call(mcp, "ingest_file", {"content": content, "file_name": "MT103.txt"})
    assert resp2.get("error") or resp2.get("result", {}).get("isError")
    text = json.dumps(resp2)
    assert "already ingested" in text
    assert "force=true" in text


def test_ingest_file_force_bypasses_duplicate_check(tmp_path):
    """force=true allows re-ingesting the same content."""
    mcp = _mcp_from_store(SQLiteStore(root=tmp_path))
    content = "Field 32A: value date, currency, amount."

    # First ingest — complete it
    resp = _call(mcp, "ingest_file", {
        "content": content,
        "file_name": "MT103.txt",
        "mode": "routing",
    })
    session_key = _parse_session_key(_text(resp))
    _complete_routing_session(mcp, session_key)

    # Second ingest with force=true — should succeed
    resp2 = _call(mcp, "ingest_file", {
        "content": content,
        "file_name": "MT103.txt",
        "force": True,
        "mode": "routing",
    })
    assert "error" not in resp2
    assert "session_key:" in _text(resp2)


def test_ingest_registry_persists_across_instances(tmp_path):
    """Registry survives MCP restart — new instance sees previous ingest."""
    from vertical_brain.storage.sqlite_store import SQLiteStore
    content = "Field 32A: value date, currency, amount."

    # First instance — ingest and complete
    store1 = SQLiteStore(root=tmp_path)
    mcp1 = _mcp_from_store(store1)
    resp = _call(mcp1, "ingest_file", {
        "content": content,
        "file_name": "spec.txt",
        "mode": "routing",
    })
    session_key = _parse_session_key(_text(resp))
    _complete_routing_session(mcp1, session_key)

    # Second instance — same DB, should see the registry entry
    store2 = SQLiteStore(root=tmp_path)
    mcp2 = _mcp_from_store(store2)
    resp2 = _call(mcp2, "ingest_file", {"content": content, "file_name": "spec.txt"})
    assert resp2.get("error") or resp2.get("result", {}).get("isError")
    assert "already ingested" in json.dumps(resp2)


def test_ingest_url_fetches_and_returns_header(tmp_path):
    """ingest_url fetches a URL, computes sha256, returns metadata header with content."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading
    import hashlib

    body = b"Field 32A: value date, currency, amount."

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence test output
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    mcp, _ = _mcp(tmp_path)
    url = f"http://127.0.0.1:{port}/docs/MT103"
    try:
        resp = _call(mcp, "ingest_url", {"url": url, "authority": "SWIFT", "doc_slug": "MT103"})
    finally:
        server.shutdown()

    text = _text(resp)
    expected_hash = hashlib.sha256(body).hexdigest()

    assert "SOURCES/MT103" in text
    assert expected_hash in text
    assert "session_key:" in text
    assert "IRON RULES" in text


def test_ingest_url_strips_html_tags(tmp_path):
    """ingest_url strips HTML tags, leaving only visible text."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    html_body = b"<html><head><title>T</title><script>x=1</script></head><body><h1>Hello</h1><p>World</p></body></html>"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    mcp, _ = _mcp(tmp_path)
    try:
        resp = _call(mcp, "ingest_url", {"url": f"http://127.0.0.1:{port}/"})
    finally:
        server.shutdown()

    text = _text(resp)
    assert "Hello" in text
    assert "World" in text
    assert "<html>" not in text
    assert "x=1" not in text             # script tag content stripped


def test_ingest_url_rejects_non_http_scheme(tmp_path):
    mcp, _ = _mcp(tmp_path)
    resp = _call(mcp, "ingest_url", {"url": "ftp://example.com/doc"})
    assert resp.get("error") or resp.get("result", {}).get("isError")


def test_ingest_url_derives_slug_from_path_segment(tmp_path):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"some content")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    mcp, _ = _mcp(tmp_path)
    try:
        resp = _call(mcp, "ingest_url", {"url": f"http://127.0.0.1:{port}/specs/my-doc"})
    finally:
        server.shutdown()

    text = _text(resp)
    assert "SOURCES/my-doc" in text
    assert "my-doc" in text              # last path segment used as slug


def test_ingest_file_force_wipes_namespace(tmp_path):
    from vertical_brain.core.models import Chunk

    mcp, store = _mcp(tmp_path)
    path = "SOURCES/MT103"
    store.ensure_node(path)
    old = Chunk(node_path=path, layer="bronze", content="old fact", immutable=True)
    store.save_chunk(old)

    resp = _call(mcp, "ingest_file", {
        "content": "Field 32A: value date, currency, amount.",
        "file_name": "MT103.txt",
        "force": True,
        "doc_slug": "MT103",
        "mode": "routing",
    })
    text = _text(resp)
    assert "force re-ingest: marked 1 prior chunk" in text
    active = [c for c in store.get_chunks_by_path(path) if c.status == "active"]
    assert all(c.id != old.id for c in active)


def test_ingest_file_force_cancels_zombie_sessions_for_namespace(tmp_path):
    """force=true removes prior ingest sessions for the same source_namespace."""
    store = SQLiteStore(root=tmp_path)
    mcp = _mcp_from_store(store)
    first = _call(mcp, "ingest_file", {
        "content": "Field 32A: value date, currency, amount.",
        "file_name": "MT103.txt",
        "doc_slug": "MT103",
        "mode": "routing",
    })
    old_session_key = _parse_session_key(_text(first))
    assert old_session_key in mcp._ingest_sessions
    assert store.get_ingest_session(old_session_key) is not None

    second = _call(mcp, "ingest_file", {
        "content": "Field 32B: updated value date, currency, amount.",
        "file_name": "MT103.txt",
        "doc_slug": "MT103",
        "mode": "routing",
        "force": True,
    })
    text = _text(second)
    new_session_key = _parse_session_key(text)
    assert "cancelled_sessions: 1" in text
    assert old_session_key not in mcp._ingest_sessions
    assert store.get_ingest_session(old_session_key) is None
    assert new_session_key in mcp._ingest_sessions
    assert store.get_ingest_session(new_session_key) is not None


def test_ingest_session_survives_mcp_restart(tmp_path):
    store = SQLiteStore(root=tmp_path)
    mcp1 = _mcp_from_store(store)
    resp = _call(mcp1, "ingest_file", {
        "content": "Field 32A: value date.",
        "file_name": "MT103.txt",
        "mode": "routing",
    })
    session_key = _parse_session_key(_text(resp))
    chunk_id = mcp1._ingest_sessions[session_key]["chunks"][0]["id"]
    _call(mcp1, "mark_service_chunk", {
        "session_key": session_key,
        "chunk_id": chunk_id,
        "status": "skipped",
        "skip_reason": _SKIP_OK,
    })

    mcp2 = _mcp_from_store(store)
    assert session_key in mcp2._ingest_sessions
    assert mcp2._ingest_sessions[session_key]["chunks"][0]["status"] == "skipped"


def test_server_module_docstring_does_not_mention_content_length():
    """Guard against re-introducing incorrect LSP-style framing docs."""
    import vertical_brain.mcp.server as server_module
    doc = server_module.__doc__ or ""
    assert "Content-Length" not in doc, (
        "server.py docstring must not mention Content-Length — "
        "the transport is newline-delimited JSON, not LSP framing."
    )
