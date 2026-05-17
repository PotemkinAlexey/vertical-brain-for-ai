"""MCP stdio server for Vertical Brain.

Protocol: JSON-RPC 2.0 over stdio, newline-delimited JSON (one object per line).
No external dependencies — pure stdlib.
"""
from __future__ import annotations

import io
import json
import sys
import traceback
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vertical_brain.storage.protocol import StorageProvider

from vertical_brain.core.context_session import ContextSession
from vertical_brain.core.embedding_router import EmbeddingRouter
from vertical_brain.core.embedding_search import EmbeddingSearch
from vertical_brain.core.json_schema import format_json_schema_errors, validate_json_schema
from vertical_brain.core.models import ChunkInput, LinkInput, StorageOperation, StorageOperationBatch
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.core.search import BrainSearch
from vertical_brain.llm.embedding import EmbeddingProvider, MockEmbeddingProvider

_PROTOCOL_VERSION = "2025-03-26"
_SERVER_VERSION = "0.1.0"

_TOOLS: list[dict[str, Any]] = [
    # ── Read / orientation ──────────────────────────────────────────────
    {
        "name": "session_start",
        "description": (
            "Return the AGENTS.md operating contract plus an orientation prompt: "
            "all namespaces, Gold summaries, chunk counts, and peer links. "
            "Call this at the start of every session."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "max_depth": {"type": "integer"},
                "summary_chars": {"type": "integer", "description": "Max Gold chars per node (default 200)"},
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
        "name": "list_chunks",
        "description": "List all chunks at a namespace path. Returns full content. Use to read what's stored.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "include_stale": {"type": "boolean"},
                "layer": {"type": "string", "enum": ["bronze", "silver", "gold"],
                          "description": "Filter by layer (omit for all layers)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_context",
        "description": (
            "Open a locked context capsule for a namespace path. "
            "Returns full chunk content, ancestor Gold summaries, and link handles."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "include_ancestors": {"type": "boolean"},
                "link_expansion": {
                    "type": "string",
                    "enum": ["handles_only", "expanded", "none"],
                },
                "max_items": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    # ── Search ──────────────────────────────────────────────────────────
    {
        "name": "search",
        "description": "Lexical full-text search across active chunks. Returns ranked results with snippets.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "root_path": {"type": "string"},
                "limit": {"type": "integer"},
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
                "threshold": {"type": "number"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "context_search",
        "description": (
            "Lexical search + locked context capsules. "
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
        "name": "context_search_semantic",
        "description": (
            "Semantic search + locked context capsules. "
            "Finds semantically similar chunks even without exact word matches."
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
                "threshold": {"type": "number"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "route",
        "description": "Find best-matching namespaces by comparing text embedding against Gold chunks.",
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
    # ── Write ───────────────────────────────────────────────────────────
    {
        "name": "append_chunk",
        "description": "Write a new chunk to a namespace. Creates the node chain if needed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "layer": {"type": "string", "enum": ["bronze", "silver", "gold"]},
                "content_type": {
                    "type": "string",
                    "enum": ["fact", "correction", "decision", "question", "note", "code", "artifact"],
                },
                "source": {"type": "string", "description": "Who wrote this, e.g. 'model', 'user'"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "append_gold_aspect",
        "description": (
            "Add a short semantic routing tag to the Gold chunk of a namespace. "
            "Returns overflow_path if a new sibling namespace was created."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "aspect": {
                    "type": "string",
                    "description": "Short search tag, ideally 30-100 characters.",
                },
            },
            "required": ["path", "aspect"],
        },
    },
    {
        "name": "create_link",
        "description": "Create a horizontal link between two namespaces.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_path": {"type": "string"},
                "target_path": {"type": "string"},
                "link_type": {"type": "string", "description": "e.g. 'peer', 'reference', 'derived_from'"},
                "reason": {"type": "string"},
            },
            "required": ["source_path", "target_path", "link_type", "reason"],
        },
    },
    {
        "name": "mark_stale",
        "description": (
            "Mark chunks as stale. Pass chunk_ids to target specific chunks, "
            "or omit to mark all active non-gold chunks at path."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific chunk IDs to mark stale. Omit to mark all active non-gold at path.",
                },
                "reason": {"type": "string", "description": "Why these chunks are stale (for audit trail)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "batch_append",
        "description": "Write multiple chunks in one call. All-or-nothing if using SQLite backend.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "layer": {"type": "string"},
                            "content_type": {"type": "string"},
                            "source": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["path", "content"],
                    },
                }
            },
            "required": ["chunks"],
        },
    },
    {
        "name": "session_end",
        "description": (
            "Persist a session summary following Bronze → Silver → Gold layering. "
            "If 'notes' is provided: writes notes as Bronze and summary as Silver. "
            "If only 'summary' is provided: writes it as Bronze (Silver promotion happens via optimizer). "
            "Optionally appends a Gold aspect."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Namespace to write the summary to"},
                "summary": {"type": "string", "description": "Refined Silver summary of what was learned or done"},
                "notes": {"type": "string", "description": "Optional raw Bronze notes — detailed facts, decisions, observations from the session"},
                "gold_aspect": {"type": "string", "description": "Optional durable insight to append to Gold"},
            },
            "required": ["path", "summary"],
        },
    },
    {
        "name": "update_silver",
        "description": (
            "Atomically rewrite the Silver summary for a namespace. "
            "Supersedes the existing Silver chunk and writes a new one. "
            "Requires current_silver_id — the ID of the active Silver chunk you read before synthesizing. "
            "Fails if Silver has changed since you read it (OCC protection). "
            "Use append_chunk(layer=silver) only for the very first Silver at a namespace."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Namespace to update"},
                "new_content": {"type": "string", "description": "Complete rewritten Silver summary"},
                "current_silver_id": {"type": "string", "description": "ID of the active Silver chunk you read"},
                "source_chunk_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional IDs of Bronze chunks incorporated into this Silver"},
                "confidence": {"type": "number", "description": "Confidence score (default: 1.0)"},
                "reasoning_summary": {"type": "string", "description": "Optional reasoning summary for the audit log"},
            },
            "required": ["path", "new_content", "current_silver_id"],
        },
    },
    {
        "name": "optimize",
        "description": (
            "Run the storage optimizer. "
            "Without path: full sweep across all namespaces — dedup, Silver compaction, decay, "
            "and cross-namespace Silver link discovery. "
            "With path: optimize that branch only, then run link discovery globally."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": [],
        },
    },
    {
        "name": "operations",
        "description": (
            "Apply a StorageOperationBatch JSON object. Validated against schema before any I/O. "
            "Supports OCC via branch_path/start_version. Returns status and per-operation results."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "description": "A StorageOperationBatch JSON object with an 'operations' array.",
                },
            },
            "required": ["payload"],
        },
    },
    {
        "name": "doctor",
        "description": "Run storage integrity checks. Returns a list of issues (orphan links, duplicates, stale staging items, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "vacuum",
        "description": (
            "Databricks-style maintenance vacuum. Dry-run by default. "
            "Physically purges old stale/superseded chunks, orphan vectors, and empty namespaces."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "retention_hours": {"type": "number", "description": "Default 168 hours"},
                "dry_run": {"type": "boolean", "description": "Default true"},
                "force": {
                    "type": "boolean",
                    "description": "Required to apply retention below 168 hours",
                },
                "prune_empty_nodes": {"type": "boolean", "description": "Default true"},
                "prune_vector_cache": {"type": "boolean", "description": "Default true"},
                "reclaim_space": {"type": "boolean", "description": "Run SQLite VACUUM after purging"},
            },
        },
    },
]

_TOOLS_BY_NAME: dict[str, dict[str, Any]] = {tool["name"]: tool for tool in _TOOLS}


class MessageParseError(ValueError):
    """Raised when stdio framing or JSON payload parsing fails."""


class VerticalBrainMCP:
    """JSON-RPC 2.0 handler. Protocol-agnostic — call handle() with parsed dicts."""

    def __init__(self, store: "StorageProvider", embedding_provider: EmbeddingProvider | None = None) -> None:
        self._store = store
        self._provider = embedding_provider or MockEmbeddingProvider()
        self._session = ContextSession(store)
        self._executor = StorageOperationExecutor(store)
        self._client_source: str = "model:mcp"  # updated on initialize

    # ------------------------------------------------------------------
    # Protocol dispatch
    # ------------------------------------------------------------------

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method", "")
        req_id = request.get("id")

        if method == "initialize":
            client_info = request.get("params", {}).get("clientInfo", {})
            client_name = client_info.get("name", "")
            if client_name:
                self._client_source = f"model:{client_name}"
            return self._reply(req_id, {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {"listChanged": False},
                },
                "serverInfo": {"name": "vertical-brain", "version": _SERVER_VERSION},
            })
        if method in ("initialized", "notifications/initialized"):
            return None  # lifecycle notification — no response
        if method.startswith("notifications/"):
            return None  # all other notifications are silently ignored
        if method == "ping":
            return self._reply(req_id, {})
        if method == "tools/list":
            return self._reply(req_id, {"tools": _TOOLS})
        if method == "tools/call":
            return self._dispatch_tool(req_id, request.get("params", {}))
        return self._error(req_id if req_id is not None else 0, -32601, f"Method not found: {method}")

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(params, dict):
            return self._error(req_id, -32602, "tools/call params must be an object")
        name = params.get("name", "")
        if name not in _TOOLS_BY_NAME:
            return self._error(req_id, -32602, f"Unknown tool: {name}")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            return self._error(req_id, -32602, "tool arguments must be an object")
        validation = validate_json_schema(args, _TOOLS_BY_NAME[name].get("inputSchema", {"type": "object"}))
        if not validation.valid:
            return self._error(req_id, -32602, f"Invalid tool arguments: {format_json_schema_errors(validation)}")
        try:
            text = self._call_tool(name, args)
            return self._reply(req_id, {"content": [{"type": "text", "text": text}]})
        except KeyError as exc:
            return self._error(req_id, -32602, f"Invalid tool arguments: missing {exc}")
        except Exception as exc:
            return self._error(req_id, -32603, f"{type(exc).__name__}: {exc}")

    def _call_tool(self, name: str, args: dict[str, Any]) -> str:  # noqa: PLR0912
        # ── Read / orientation ──────────────────────────────────────────
        if name == "session_start":
            return self._session.session_prompt(
                root_path=args.get("root_path"),
                max_depth=args.get("max_depth"),
                summary_max_chars=args.get("summary_chars", 200),
            )

        if name == "namespace_map":
            return self._session.namespace_map(
                root_path=args.get("root_path"),
                max_depth=args.get("max_depth"),
                summary_max_chars=args.get("summary_chars", 240),
            ).to_json()

        if name == "list_chunks":
            chunks = self._store.get_chunks_by_path(args["path"])  # type: ignore[attr-defined]
            if not args.get("include_stale", False):
                chunks = [c for c in chunks if c.status == "active"]
            if "layer" in args:
                chunks = [c for c in chunks if c.layer == args["layer"]]
            return json.dumps([
                {"id": c.id, "layer": c.layer, "content_type": c.content_type,
                 "status": c.status, "source": c.source, "confidence": c.confidence,
                 "content": c.content,
                 "created_at": c.created_at.isoformat() if hasattr(c.created_at, "isoformat") else str(c.created_at)}
                for c in chunks
            ], ensure_ascii=False, indent=2)

        if name == "read_context":
            from vertical_brain.core.context_lock import ContextLock
            from vertical_brain.core.models import ContextBudget, ContextPolicy
            policy = ContextPolicy(
                include_ancestors=args.get("include_ancestors", True),
                include_target=True,
                link_expansion=args.get("link_expansion", "handles_only"),
            )
            budget = ContextBudget(max_items=args.get("max_items", 20))
            locked = ContextLock(self._store).open_locked_context(  # type: ignore[arg-type]
                args["path"], policy=policy, budget=budget
            )
            return locked.to_json()

        # ── Search ──────────────────────────────────────────────────────
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

        if name == "context_search_semantic":
            result = self._session.search_locked_context_semantic(
                args["query"],
                self._provider,
                root_path=args.get("root_path"),
                search_limit=args.get("search_limit", 10),
                context_limit=args.get("context_limit", 3),
                items_per_context=args.get("items_per_context", 6),
                include_ancestors=args.get("include_ancestors", True),
                threshold=args.get("threshold", 0.0),
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

        # ── Write ───────────────────────────────────────────────────────
        if name == "append_chunk":
            op = StorageOperation(
                operation="append_chunk",
                target_path=args["path"],
                chunk=ChunkInput(
                    content=args["content"],
                    layer=args.get("layer", "bronze"),
                    content_type=args.get("content_type", "fact"),
                    source=args.get("source", self._client_source),
                    confidence=args.get("confidence", 1.0),
                ),
                reasoning_summary=args.get("reasoning_summary", "Appended via MCP append_chunk."),
            )
            result = self._executor.apply(op)
            return json.dumps({"chunk_id": result.chunk_id, "status": result.status})

        if name == "append_gold_aspect":
            op = StorageOperation(
                operation="append_gold_aspect",
                target_path=args["path"],
                gold_aspect=args["aspect"],
                reasoning_summary=args.get("reasoning_summary", "Updated Gold aspect via MCP."),
            )
            result = self._executor.apply(op)
            out: dict[str, Any] = {"status": result.status, "path": result.target_path}
            if result.overflow_path:
                out["overflow_path"] = result.overflow_path
            if result.aspect_too_long:
                out["aspect_too_long"] = True
            return json.dumps(out)

        if name == "create_link":
            op = StorageOperation(
                operation="create_link",
                target_path=args["source_path"],
                links=[LinkInput(
                    target_path=args["target_path"],
                    link_type=args["link_type"],
                    reason=args["reason"],
                )],
                reasoning_summary=args.get("reasoning_summary", "Created link via MCP."),
            )
            result = self._executor.apply(op)
            return json.dumps({"link_ids": result.link_ids, "status": result.status})

        if name == "mark_stale":
            chunk_ids: list[str] = args.get("chunk_ids") or []
            if not chunk_ids:
                # Mark all active non-gold chunks at path
                chunks = self._store.get_chunks_by_path(args["path"])  # type: ignore[attr-defined]
                chunk_ids = [
                    c.id for c in chunks
                    if c.status == "active" and c.layer != "gold"
                ]
            if not chunk_ids:
                return json.dumps({"status": "applied", "marked": 0})
            op = StorageOperation(
                operation="mark_stale",
                target_path=args["path"],
                chunk_ids=chunk_ids,
                reasoning_summary=args.get("reason", "Marked stale via MCP."),
            )
            self._executor.apply(op)
            return json.dumps({"status": "applied", "marked": len(chunk_ids)})

        if name == "batch_append":
            chunks_data: list[dict[str, Any]] = args.get("chunks", [])
            if not chunks_data:
                return json.dumps({"status": "applied", "chunk_ids": []})
            batch = StorageOperationBatch(
                operations=[
                    StorageOperation(
                        operation="append_chunk",
                        target_path=item["path"],
                        chunk=ChunkInput(
                            content=item["content"],
                            layer=item.get("layer", "bronze"),
                            content_type=item.get("content_type", "fact"),
                            source=item.get("source", self._client_source),
                            confidence=item.get("confidence", 1.0),
                        ),
                    )
                    for item in chunks_data
                ],
                reasoning_summary=args.get("reasoning_summary", "Batch appended via MCP batch_append."),
            )
            batch_result = self._executor.apply_batch(batch)
            chunk_ids_written = [r.chunk_id for r in batch_result.results if r.chunk_id]
            return json.dumps({"status": batch_result.status, "chunk_ids": chunk_ids_written})

        if name == "session_end":
            notes = args.get("notes")
            summary = args["summary"]
            ops: list[StorageOperation] = []
            # Bronze: raw notes if provided, otherwise summary is the raw capture
            ops.append(StorageOperation(
                operation="append_chunk",
                target_path=args["path"],
                chunk=ChunkInput(
                    content=notes if notes else summary,
                    layer="bronze",
                    content_type="note",
                    source=self._client_source,
                ),
                reasoning_summary="Persisted session Bronze notes.",
            ))
            # Silver: when notes and summary are distinct, OR when gold_aspect requires it
            if notes or args.get("gold_aspect"):
                ops.append(StorageOperation(
                    operation="append_chunk",
                    target_path=args["path"],
                    chunk=ChunkInput(
                        content=summary,
                        layer="silver",
                        content_type="note",
                        source=self._client_source,
                    ),
                    reasoning_summary="Persisted session Silver summary.",
                ))
            batch = StorageOperationBatch(
                operations=ops,
                reasoning_summary="session_end: Bronze → Silver layering.",
            )
            batch_result = self._executor.apply_batch(batch)
            chunk_ids = [r.chunk_id for r in batch_result.results if r.chunk_id]
            out = {"status": batch_result.status, "chunk_ids": chunk_ids}
            if args.get("gold_aspect"):
                gold_op = StorageOperation(
                    operation="append_gold_aspect",
                    target_path=args["path"],
                    gold_aspect=args["gold_aspect"],
                    reasoning_summary="Updated Gold aspect via session_end.",
                )
                gold_result = self._executor.apply(gold_op)
                out["gold_status"] = gold_result.status
                if gold_result.overflow_path:
                    out["overflow_path"] = gold_result.overflow_path
                if gold_result.aspect_too_long:
                    out["aspect_too_long"] = True
            return json.dumps(out)

        if name == "update_silver":
            op = StorageOperation(
                operation="update_silver",
                target_path=args["path"],
                current_silver_id=args["current_silver_id"],
                source_chunk_ids=args.get("source_chunk_ids") or [],
                chunk=ChunkInput(
                    content=args["new_content"],
                    layer="silver",
                    content_type="note",
                    source=self._client_source,
                    confidence=args.get("confidence", 1.0),
                ),
                reasoning_summary=args.get("reasoning_summary") or "Updated Silver summary via MCP update_silver.",
            )
            result = self._executor.apply(op)
            return json.dumps({"status": "applied", "chunk_id": result.chunk_id})

        if name == "optimize":
            from vertical_brain.core.optimizer import SimpleOptimizer
            from vertical_brain.core.router import StorageModel
            from pathlib import Path
            model_file = Path(__file__).resolve().parents[3] / "data" / "namespaces" / "model.json"
            model = StorageModel.load(str(model_file)) if model_file.exists() else None
            min_parts = model.min_compaction_path_parts if model else 3
            optimizer = SimpleOptimizer(
                self._store,  # type: ignore[arg-type]
                min_compaction_path_parts=min_parts,
                embedding_provider=self._provider,
            )
            path = args.get("path")
            if path:
                branch_result = optimizer.optimize_branch(path)
                link_result = optimizer.discover_links()
                result = branch_result + "\n\n" + link_result
            else:
                result = optimizer.optimize_all()
            return json.dumps({"status": "applied", "result": str(result)})

        if name == "operations":
            from vertical_brain.core.operations import operation_batch_from_dict
            payload = args["payload"]
            if not isinstance(payload, dict):
                raise ValueError("payload must be a JSON object")
            batch = operation_batch_from_dict(payload)
            batch_result = self._executor.apply_batch(batch)
            return batch_result.to_json()

        if name == "doctor":
            from vertical_brain.core.doctor import Doctor
            issues = Doctor(self._store).run()  # type: ignore[arg-type]
            return json.dumps([
                {"severity": i.severity, "check": i.check, "message": i.message, "path": i.path}
                for i in issues
            ], ensure_ascii=False, indent=2)

        if name == "vacuum":
            vacuum = getattr(self._store, "vacuum", None)
            if not callable(vacuum):
                raise ValueError("vacuum is only supported by the SQLite storage backend")
            result = vacuum(
                retention_hours=args.get("retention_hours", 168.0),
                dry_run=args.get("dry_run", True),
                force=args.get("force", False),
                prune_empty_nodes=args.get("prune_empty_nodes", True),
                prune_vector_cache=args.get("prune_vector_cache", True),
                reclaim_space=args.get("reclaim_space", False),
            )
            return json.dumps(result, ensure_ascii=False, indent=2)

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
    """Read one newline-delimited JSON message from *stream*.

    MCP stdio transport sends one JSON object per line (newline-delimited JSON).
    Empty lines are skipped so the reader is tolerant of blank separators.
    """
    while True:
        line = stream.readline()
        if not line:
            raise EOFError
        try:
            decoded = line.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise MessageParseError("Message line is not valid UTF-8") from exc
        if not decoded:
            continue  # skip blank lines
        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise MessageParseError(f"Invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise MessageParseError("Message must be a JSON object")
        return payload


def _write_message(stream: io.RawIOBase, obj: dict[str, Any]) -> None:
    """Write one newline-delimited JSON message to *stream*."""
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    stream.write(line.encode("utf-8"))
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
        except MessageParseError as exc:
            _write_message(stdout, VerticalBrainMCP._error(0, -32700, str(exc)))
            break
        except Exception:
            traceback.print_exc(file=sys.stderr)
            break

        try:
            response = server.handle(request)
        except Exception as exc:
            req_id = request.get("id")
            response = VerticalBrainMCP._error(req_id if req_id is not None else 0, -32603, str(exc))

        if response is not None:
            _write_message(stdout, response)
