"""Tests for the v1.13 MCP middleware hooks (before_tool / after_tool).

These hooks are how enterprise plugs in RBAC, rate-limiting, scoping,
audit decoration, billing, and request correlation without forking the
open core. Default implementations are pass-through — overriding them is
the supported extension path.
"""
from __future__ import annotations

import json

from vertical_brain.core.models import Chunk
from vertical_brain.mcp.server import ToolAccessDenied, VerticalBrainMCP
from vertical_brain.storage.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Default behaviour (no overrides) — round-trip
# ---------------------------------------------------------------------------


def test_default_before_after_tool_pass_through(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="alpha one"))
    server = VerticalBrainMCP(store)

    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "search", "arguments": {"query": "alpha"}},
    })

    assert "error" not in response
    data = json.loads(response["result"]["content"][0]["text"])
    assert any(r["path"] == "WORK/A" for r in data)


# ---------------------------------------------------------------------------
# before_tool: arg rewriting
# ---------------------------------------------------------------------------


class _ScopingServer(VerticalBrainMCP):
    """Enterprise pattern: inject `root_path` scoping from a tenant claim."""

    def __init__(self, store, tenant_root: str) -> None:
        super().__init__(store)
        self._tenant_root = tenant_root
        self.calls: list[tuple[str, dict]] = []

    def _before_tool(self, name, args, *, request_id):
        self.calls.append((name, dict(args)))
        if name in {"search", "search_semantic", "route", "context_search", "context_search_semantic"}:
            args.setdefault("root_path", self._tenant_root)
        return args


def test_before_tool_can_inject_root_path(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="ACME/A", content="alpha"))
    store.save_chunk(Chunk(node_path="GLOBEX/A", content="alpha"))
    server = _ScopingServer(store, tenant_root="ACME")

    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "search", "arguments": {"query": "alpha"}},
    })

    data = json.loads(response["result"]["content"][0]["text"])
    paths = {r["path"] for r in data}
    assert "ACME/A" in paths
    assert "GLOBEX/A" not in paths
    # Hook saw the original args (no root_path) and the call name.
    assert server.calls[0] == ("search", {"query": "alpha"})


def test_before_tool_args_are_isolated_per_call(tmp_path):
    """Mutations made by _before_tool to the local args dict must not leak
    back to the JSON-RPC params on subsequent calls."""
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="ACME/A", content="alpha"))
    server = _ScopingServer(store, tenant_root="ACME")
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "search", "arguments": {"query": "alpha"}},
    }
    server.handle(payload)
    # Subsequent call without a root_path — should still get scoping injected.
    server.handle(payload)
    assert all(c[1] == {"query": "alpha"} for c in server.calls)


# ---------------------------------------------------------------------------
# before_tool: ToolAccessDenied
# ---------------------------------------------------------------------------


class _ACLServer(VerticalBrainMCP):
    def _before_tool(self, name, args, *, request_id):
        if name == "ingest_file":
            raise ToolAccessDenied("ingest is disabled for tenant", code=-32010)
        return args


def test_before_tool_can_reject_with_tool_access_denied(tmp_path):
    store = SQLiteStore(tmp_path)
    server = _ACLServer(store)

    response = server.handle({
        "jsonrpc": "2.0", "id": 7, "method": "tools/call",
        "params": {"name": "ingest_file", "arguments": {"path": "x.txt", "target_namespace": "WORK"}},
    })

    assert response.get("error") is not None
    assert response["error"]["code"] == -32010
    assert "ingest is disabled" in response["error"]["message"]


def test_tool_access_denied_default_code_is_application_denial():
    exc = ToolAccessDenied("nope")
    assert exc.code == -32004


# ---------------------------------------------------------------------------
# before_tool: invalid return value
# ---------------------------------------------------------------------------


class _BadBeforeServer(VerticalBrainMCP):
    def _before_tool(self, name, args, *, request_id):
        return "not a dict"  # type: ignore[return-value]


def test_before_tool_must_return_dict(tmp_path):
    store = SQLiteStore(tmp_path)
    server = _BadBeforeServer(store)

    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "search", "arguments": {"query": "x"}},
    })

    assert response.get("error") is not None
    assert response["error"]["code"] == -32603
    assert "_before_tool must return a dict" in response["error"]["message"]


# ---------------------------------------------------------------------------
# after_tool: response decoration
# ---------------------------------------------------------------------------


class _DecoratingServer(VerticalBrainMCP):
    """Enterprise pattern: attach an audit marker to every response."""

    def _after_tool(self, name, args, response, *, request_id):
        if "result" in response and isinstance(response["result"], dict):
            content_blocks = response["result"].get("content", [])
            if content_blocks and isinstance(content_blocks[0], dict):
                content_blocks[0].setdefault("annotations", {})["audit_id"] = f"req-{request_id}"
        return response


def test_after_tool_can_decorate_response(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="A", content="x"))
    server = _DecoratingServer(store)

    response = server.handle({
        "jsonrpc": "2.0", "id": 42, "method": "tools/call",
        "params": {"name": "search", "arguments": {"query": "x"}},
    })

    assert "error" not in response
    annotations = response["result"]["content"][0].get("annotations", {})
    assert annotations.get("audit_id") == "req-42"


# ---------------------------------------------------------------------------
# after_tool: failure surface
# ---------------------------------------------------------------------------


class _ExplodingAfterServer(VerticalBrainMCP):
    def _after_tool(self, name, args, response, *, request_id):
        raise RuntimeError("post-processing pipeline crashed")


def test_after_tool_exception_surfaces_as_internal_error(tmp_path):
    store = SQLiteStore(tmp_path)
    server = _ExplodingAfterServer(store)

    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "search", "arguments": {"query": "x"}},
    })

    assert response.get("error") is not None
    assert response["error"]["code"] == -32603
    assert "_after_tool failed" in response["error"]["message"]


# ---------------------------------------------------------------------------
# Hook ordering: before runs before validation; after runs after tool body
# ---------------------------------------------------------------------------


class _OrderTrackingServer(VerticalBrainMCP):
    def __init__(self, store) -> None:
        super().__init__(store)
        self.events: list[str] = []

    def _before_tool(self, name, args, *, request_id):
        self.events.append(f"before:{name}")
        return args

    def _after_tool(self, name, args, response, *, request_id):
        self.events.append(f"after:{name}")
        return response


def test_hook_ordering_before_then_after(tmp_path):
    store = SQLiteStore(tmp_path)
    server = _OrderTrackingServer(store)

    server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "namespace_map", "arguments": {}},
    })

    assert server.events == ["before:namespace_map", "after:namespace_map"]


def test_hook_ordering_before_runs_before_validation(tmp_path):
    """before_tool gets a chance to mutate args BEFORE schema validation —
    so it can supply a missing required field."""

    class _SupplyMissingArg(VerticalBrainMCP):
        def _before_tool(self, name, args, *, request_id):
            if name == "search":
                args.setdefault("query", "alpha")
            return args

    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="A", content="alpha"))
    server = _SupplyMissingArg(store)

    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "search", "arguments": {}},
    })
    assert "error" not in response


def test_hook_ordering_after_skipped_on_tool_error(tmp_path):
    """When the tool body raises, after_tool MUST NOT run — the response is
    already an error reply."""

    class _RecordAfter(VerticalBrainMCP):
        def __init__(self, store) -> None:
            super().__init__(store)
            self.after_called = False
        def _after_tool(self, name, args, response, *, request_id):
            self.after_called = True
            return response

    store = SQLiteStore(tmp_path)
    server = _RecordAfter(store)

    # Trigger a tool body error: ingest_file with a path that does not exist.
    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {
            "name": "ingest_file",
            "arguments": {"path": "/no/such/file.txt", "target_namespace": "WORK"},
        },
    })
    assert response.get("error") is not None
    assert server.after_called is False
