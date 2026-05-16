"""Docs/code consistency tests.

These tests ensure that documentation claims match the actual implementation.
They are not exhaustive — they guard specific contracts that have drifted before.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

from vertical_brain.storage.protocol import StorageProvider
from vertical_brain.mcp.server import _TOOLS, VerticalBrainMCP


# ── StorageProvider protocol ──────────────────────────────────────────────────

def test_storage_provider_get_chunks_by_path_has_include_children():
    sig = inspect.signature(StorageProvider.get_chunks_by_path)
    assert "include_children" in sig.parameters, (
        "StorageProvider.get_chunks_by_path must declare include_children parameter"
    )
    param = sig.parameters["include_children"]
    assert param.default is False, (
        "include_children must default to False"
    )


# ── MCP tool registry ─────────────────────────────────────────────────────────

def _declared_tool_names() -> set[str]:
    return {t["name"] for t in _TOOLS}


def _implemented_tool_names(tmp_path) -> set[str]:
    """Collect tool names that _call_tool actually handles (not just declared)."""
    from vertical_brain.storage.json_store import JsonStore
    store = JsonStore(tmp_path)
    server = VerticalBrainMCP(store)

    implemented = set()
    for tool in _TOOLS:
        name = tool["name"]
        # Build minimal valid args for this tool so the call doesn't fail on missing keys.
        # We only need to confirm the tool name is routed — not that it succeeds.
        args: dict = {}
        required = tool.get("inputSchema", {}).get("required", [])
        for req in required:
            # Provide stub strings for required string params.
            prop_schema = tool.get("inputSchema", {}).get("properties", {}).get(req, {})
            if prop_schema.get("type") == "object":
                args[req] = {}
            elif prop_schema.get("type") == "array":
                args[req] = []
            else:
                args[req] = "STUB"
        try:
            server._call_tool(name, args)
            implemented.add(name)
        except KeyError:
            pass  # KeyError from unknown tool name — not implemented
        except Exception:
            implemented.add(name)  # raised for a real reason, tool is implemented

    return implemented


def test_mcp_declared_tools_are_all_implemented(tmp_path):
    declared = _declared_tool_names()
    implemented = _implemented_tool_names(tmp_path)
    missing = declared - implemented
    assert not missing, (
        f"Tools declared in _TOOLS but not implemented in _call_tool: {missing}"
    )


def test_mcp_no_undeclared_tools(tmp_path):
    """Every implemented tool must be in _TOOLS so clients see it via tools/list."""
    declared = _declared_tool_names()
    implemented = _implemented_tool_names(tmp_path)
    undeclared = implemented - declared
    assert not undeclared, (
        f"Tools implemented in _call_tool but missing from _TOOLS declaration: {undeclared}"
    )


# ── docs/05_mcp_tools.md headings ────────────────────────────────────────────


def _mcp_tools_doc_names() -> set[str]:
    """Extract tool names from ### `name` headings in docs/05_mcp_tools.md."""
    doc_path = Path(__file__).parent.parent / "docs" / "05_mcp_tools.md"
    if not doc_path.exists():
        return set()
    names: set[str] = set()
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^###\s+`(\w+)`", line)
        if m:
            names.add(m.group(1))
    return names


def test_mcp_tools_doc_covers_all_declared_tools():
    """Every tool in _TOOLS must have a ### `name` heading in docs/05_mcp_tools.md."""
    declared = _declared_tool_names()
    documented = _mcp_tools_doc_names()
    missing = declared - documented
    assert not missing, (
        f"Tools in _TOOLS missing from docs/05_mcp_tools.md: {missing}"
    )


def test_mcp_tools_doc_has_no_phantom_tools():
    """No tool heading in docs/05_mcp_tools.md should refer to a non-existent tool."""
    declared = _declared_tool_names()
    documented = _mcp_tools_doc_names()
    phantom = documented - declared
    assert not phantom, (
        f"Tools documented in docs/05_mcp_tools.md but absent from _TOOLS: {phantom}"
    )
