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
        "operations", "doctor", "vacuum", "ingest_file",
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
    assert result["aspect_too_long"] is True


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


def test_ingest_file_returns_prompt_with_content(tmp_path):
    """ingest_file returns compact metadata header with embedded file content."""
    mcp, _ = _mcp(tmp_path)
    sample = tmp_path / "MT103.txt"
    sample.write_text("Field 32A: Value Date, Currency, Amount. Format: 6!n3!a15d")

    resp = _call(mcp, "ingest_file", {
        "file_path": str(sample),
        "authority": "SWIFT",
        "doc_slug": "MT103",
    })
    text = _text(resp)

    assert "MT103.txt" in text
    assert "SOURCES/SWIFT/MT103" in text
    assert "Field 32A" in text        # file content embedded
    assert "sha256" in text.lower()   # hash present
    assert "AGENTS.md" in text        # references the universal protocol


def test_ingest_file_defaults_slug_to_filename(tmp_path):
    mcp, _ = _mcp(tmp_path)
    sample = tmp_path / "my_spec.md"
    sample.write_text("# Spec\nSome content here.")

    resp = _call(mcp, "ingest_file", {"file_path": str(sample)})
    text = _text(resp)

    assert "my_spec" in text          # slug derived from filename


def test_ingest_file_raises_on_missing_file(tmp_path):
    mcp, _ = _mcp(tmp_path)
    resp = _call(mcp, "ingest_file", {"file_path": str(tmp_path / "nonexistent.txt")})
    assert resp.get("error") or (resp.get("result", {}).get("isError"))


def test_ingest_file_hash_is_sha256(tmp_path):
    import hashlib
    mcp, _ = _mcp(tmp_path)
    content = "deterministic content for hashing"
    sample = tmp_path / "doc.txt"
    sample.write_text(content)
    expected_hash = hashlib.sha256(content.encode()).hexdigest()

    resp = _call(mcp, "ingest_file", {"file_path": str(sample)})
    text = _text(resp)

    assert expected_hash in text


def test_server_module_docstring_does_not_mention_content_length():
    """Guard against re-introducing incorrect LSP-style framing docs."""
    import vertical_brain.mcp.server as server_module
    doc = server_module.__doc__ or ""
    assert "Content-Length" not in doc, (
        "server.py docstring must not mention Content-Length — "
        "the transport is newline-delimited JSON, not LSP framing."
    )
