"""MCP stdio server for Vertical Brain.

Protocol: JSON-RPC 2.0 over stdio with Content-Length framing (LSP-style).
No external dependencies — pure stdlib.
"""
from __future__ import annotations

import io
import json
import sys
import traceback
from typing import Any

from vertical_brain.core.context_session import ContextSession
from vertical_brain.core.embedding_router import EmbeddingRouter
from vertical_brain.core.embedding_search import EmbeddingSearch
from vertical_brain.core.models import Chunk, ChunkInput, StorageOperation
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.core.search import BrainSearch
from vertical_brain.llm.embedding import EmbeddingProvider, MockEmbeddingProvider

_PROTOCOL_VERSION = "2024-11-05"
_SERVER_VERSION = "0.1.0"

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "session_start",
        "description": (
            "Return a compact orientation prompt describing all namespaces, "
            "Gold summaries, chunk counts, and peer links. Call this at the "
            "start of every session to recover context."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string", "description": "Limit to this namespace branch"},
                "max_depth": {"type": "integer", "description": "Maximum depth relative to root_path"},
                "summary_chars": {"type": "integer", "description": "Max Gold summary chars per node (default 200)"},
            },
        },
    },
    {
        "name": "namespace_map",
        "description": "Return the namespace map as JSON (nodes, counts, Gold summaries, link handles).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "max_depth": {"type": "integer"},
                "summary_chars": {"type": "integer"},
            },
        },
    },
    {
        "name": "search",
        "description": "Lexical full-text search across active chunks. Returns ranked results with snippets.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "root_path": {"type": "string"},
                "limit": {"type": "integer", "description": "Max results (default 10)"},
                "include_stale": {"type": "boolean"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_semantic",
        "description": "Semantic similarity search across active chunks using embeddings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "root_path": {"type": "string"},
                "limit": {"type": "integer"},
                "threshold": {"type": "number", "description": "Minimum cosine similarity (0–1)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "context_search",
        "description": (
            "Find the most relevant locked context capsules for a query. "
            "Returns full chunk content for each matched namespace."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "root_path": {"type": "string"},
                "search_limit": {"type": "integer"},
                "context_limit": {"type": "integer"},
                "items_per_context": {"type": "integer"},
                "include_ancestors": {"type": "boolean"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "route",
        "description": (
            "Find the best-matching namespaces for a piece of text by comparing "
            "its embedding against Gold chunk embeddings. Use before append_chunk "
            "to pick the right namespace."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "limit": {"type": "integer"},
                "threshold": {"type": "number"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "append_chunk",
        "description": "Write a new chunk to a namespace. Creates the node chain if needed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Namespace path, e.g. WORK/DataArt/Databricks"},
                "content": {"type": "string"},
                "layer": {
                    "type": "string",
                    "enum": ["bronze", "silver", "gold"],
                    "description": "bronze=raw fact, silver=synthesized, gold=semantic label",
                },
                "content_type": {
                    "type": "string",
                    "enum": ["fact", "correction", "decision", "question", "note", "code", "artifact"],
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "append_gold_aspect",
        "description": (
            "Add a semantic label (aspect) to the Gold chunk of a namespace. "
            "Aspects accumulate as 'tag1 | tag2 | tag3'. Creates overflow sibling "
            "namespace when the Gold chunk exceeds capacity."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "aspect": {"type": "string", "description": "Short semantic label, e.g. 'Delta Lake migration'"},
            },
            "required": ["path", "aspect"],
        },
    },
]


class VerticalBrainMCP:
    """JSON-RPC 2.0 handler. Protocol-agnostic — call handle() with parsed dicts."""

    def __init__(self, store: object, embedding_provider: EmbeddingProvider | None = None) -> None:
        self._store = store
        self._provider = embedding_provider or MockEmbeddingProvider()
        self._session = ContextSession(store)  # type: ignore[arg-type]
        self._executor = StorageOperationExecutor(store)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Protocol dispatch
    # ------------------------------------------------------------------

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method", "")
        req_id = request.get("id")

        if method == "initialize":
            return self._reply(req_id, {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "vertical-brain", "version": _SERVER_VERSION},
            })

        if method == "initialized":
            return None  # notification, no response

        if method == "ping":
            return self._reply(req_id, {})

        if method == "tools/list":
            return self._reply(req_id, {"tools": _TOOLS})

        if method == "tools/call":
            return self._dispatch_tool(req_id, request.get("params", {}))

        return self._error(req_id, -32601, f"Method not found: {method}")

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name", "")
        args: dict[str, Any] = params.get("arguments") or {}
        try:
            text = self._call_tool(name, args)
            return self._reply(req_id, {"content": [{"type": "text", "text": text}]})
        except KeyError as exc:
            return self._error(req_id, -32602, f"Unknown tool: {name} ({exc})")
        except Exception as exc:
            return self._error(req_id, -32603, f"{type(exc).__name__}: {exc}")

    def _call_tool(self, name: str, args: dict[str, Any]) -> str:
        if name == "session_start":
            return self._session.session_prompt(
                root_path=args.get("root_path"),
                max_depth=args.get("max_depth"),
                summary_max_chars=args.get("summary_chars", 200),
            )

        if name == "namespace_map":
            nm = self._session.namespace_map(
                root_path=args.get("root_path"),
                max_depth=args.get("max_depth"),
                summary_max_chars=args.get("summary_chars", 240),
            )
            return nm.to_json()

        if name == "search":
            results = BrainSearch(self._store).search(  # type: ignore[arg-type]
                args["query"],
                root_path=args.get("root_path"),
                limit=args.get("limit", 10),
                include_stale=args.get("include_stale", False),
            )
            return json.dumps([
                {"path": r.path, "score": r.score, "layer": r.layer,
                 "content_type": r.content_type, "snippet": r.snippet}
                for r in results
            ], ensure_ascii=False, indent=2)

        if name == "search_semantic":
            results = EmbeddingSearch(self._store, self._provider).search(
                args["query"],
                root_path=args.get("root_path"),
                limit=args.get("limit", 10),
                threshold=args.get("threshold", 0.0),
            )
            return json.dumps([
                {"path": r.path, "score": round(r.score, 4), "layer": r.layer,
                 "content_type": r.content_type, "snippet": r.snippet}
                for r in results
            ], ensure_ascii=False, indent=2)

        if name == "context_search":
            result = self._session.search_locked_context(
                args["query"],
                root_path=args.get("root_path"),
                search_limit=args.get("search_limit", 10),
                context_limit=args.get("context_limit", 3),
                items_per_context=args.get("items_per_context", 6),
                include_ancestors=args.get("include_ancestors", True),
            )
            return result.to_json()

        if name == "route":
            candidates = EmbeddingRouter(self._store, self._provider).find_candidates(
                args["text"],
                threshold=args.get("threshold", 0.0),
                limit=args.get("limit", 5),
            )
            return json.dumps([
                {"path": c.path, "score": round(c.score, 4), "gold_summary": c.gold_summary}
                for c in candidates
            ], ensure_ascii=False, indent=2)

        if name == "append_chunk":
            op = StorageOperation(
                operation="append_chunk",
                target_path=args["path"],
                chunk=ChunkInput(
                    content=args["content"],
                    layer=args.get("layer", "bronze"),
                    content_type=args.get("content_type", "fact"),
                ),
            )
            result = self._executor.apply(op)
            return json.dumps({"chunk_id": result.chunk_id, "status": result.status})

        if name == "append_gold_aspect":
            op = StorageOperation(
                operation="append_gold_aspect",
                target_path=args["path"],
                gold_aspect=args["aspect"],
            )
            result = self._executor.apply(op)
            return json.dumps({"status": result.status, "path": result.target_path})

        raise KeyError(name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reply(req_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


# ------------------------------------------------------------------
# stdio transport
# ------------------------------------------------------------------

def _read_message(stream: io.RawIOBase) -> dict[str, Any]:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            raise EOFError
        decoded = line.decode("utf-8")
        if decoded in ("\r\n", "\n"):
            break
        key, _, value = decoded.partition(":")
        headers[key.strip()] = value.strip()
    length = int(headers["Content-Length"])
    body = stream.read(length)
    return json.loads(body.decode("utf-8"))


def _write_message(stream: io.RawIOBase, obj: dict[str, Any]) -> None:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    stream.write(header + body)
    stream.flush()


def run_stdio(store: object, embedding_provider: EmbeddingProvider | None = None) -> None:
    """Run the MCP server on stdin/stdout until EOF."""
    server = VerticalBrainMCP(store, embedding_provider)
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        try:
            request = _read_message(stdin)
        except EOFError:
            break
        except Exception:
            traceback.print_exc(file=sys.stderr)
            break

        try:
            response = server.handle(request)
        except Exception as exc:
            response = VerticalBrainMCP._error(request.get("id"), -32603, str(exc))

        if response is not None:
            _write_message(stdout, response)
