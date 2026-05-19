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
from vertical_brain.mcp.ingest_handlers import _IngestHandlers
from vertical_brain.mcp.tools import _TOOLS, _TOOLS_BY_NAME, root_path_from_args

_PROTOCOL_VERSION = "2025-03-26"
_SERVER_VERSION = "0.1.0"


class MessageParseError(ValueError):
    """Raised when stdio framing or JSON payload parsing fails."""


class VerticalBrainMCP(_IngestHandlers):
    """JSON-RPC 2.0 handler. Protocol-agnostic — call handle() with parsed dicts."""

    def __init__(self, store: "StorageProvider", embedding_provider: EmbeddingProvider | None = None) -> None:
        self._store = store
        self._provider = embedding_provider or MockEmbeddingProvider()
        self._session = ContextSession(store)
        self._executor = StorageOperationExecutor(store)
        self._client_source: str = "model:mcp"  # updated on initialize
        self._ingest_sessions: dict[str, dict[str, Any]] = {}
        self._load_ingest_sessions_from_store()


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
                root_path=root_path_from_args(args),
                max_depth=args.get("max_depth"),
                summary_max_chars=args.get("summary_chars", 200),
            )

        if name == "namespace_map":
            return self._session.namespace_map(
                root_path=root_path_from_args(args),
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
                root_path=root_path_from_args(args),
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
                root_path=root_path_from_args(args),
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
                root_path=root_path_from_args(args),
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
                root_path=root_path_from_args(args),
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
                    immutable=bool(args.get("immutable", False)),
                ),
                reasoning_summary=args.get("reasoning_summary", "Appended via MCP append_chunk."),
            )
            result = self._executor.apply(op)
            out: dict[str, Any] = {"chunk_id": result.chunk_id, "status": result.status}
            if result.similar_bronze:
                out["similar_bronze"] = [
                    {"chunk_id": s.chunk_id, "score": s.score, "snippet": s.snippet[:120]}
                    for s in result.similar_bronze
                ]
            if result.silver_too_large:
                out["silver_too_large"] = True
            text = json.dumps(out)
            layer = args.get("layer", "bronze")
            if layer == "bronze":
                text += (
                    "\n\n---\nAGENTS: Bronze written → update Silver now (update_silver). "
                    "Write after every decision or code change — not only at session_end."
                )
            return text

        if name == "append_gold_aspect":
            aspects: list[str] = []
            if args.get("aspect"):
                aspects.append(args["aspect"])
            aspects.extend(args.get("aspects") or [])
            if not aspects:
                raise ValueError("append_gold_aspect requires 'aspect' or 'aspects'")
            reasoning = args.get("reasoning_summary", "Updated Gold aspect via MCP.")
            batch = StorageOperationBatch(
                operations=[
                    StorageOperation(
                        operation="append_gold_aspect",
                        target_path=args["path"],
                        gold_aspect=aspect,
                    )
                    for aspect in aspects
                ],
                reasoning_summary=reasoning,
            )
            batch_result = self._executor.apply_batch(batch)
            out2: dict[str, Any] = {
                "status": batch_result.status,
                "path": args["path"],
                "aspects_added": len(aspects),
            }
            overflow_paths = [r.overflow_path for r in batch_result.results if r.overflow_path]
            too_long = [
                aspect for aspect, r in zip(aspects, batch_result.results) if r.aspect_too_long
            ]
            if overflow_paths:
                out2["overflow_paths"] = overflow_paths
            if too_long:
                out2["aspects_too_long"] = too_long
            if any(r.gold_near_limit for r in batch_result.results):
                out2["gold_near_limit"] = True
            return json.dumps(out2)

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
            from collections import defaultdict
            include_children = bool(args.get("recursive", False))
            include_immutable = bool(args.get("include_immutable", False))
            explicit_ids: list[str] = args.get("chunk_ids") or []

            if explicit_ids:
                all_chunks = self._store.get_chunks_by_path(  # type: ignore[attr-defined]
                    args["path"], include_children=True
                )
                id_to_path = {c.id: c.node_path for c in all_chunks}
                chunk_ids = explicit_ids
            else:
                all_chunks = self._store.get_chunks_by_path(  # type: ignore[attr-defined]
                    args["path"], include_children=include_children
                )
                id_to_path = {c.id: c.node_path for c in all_chunks}
                chunk_ids = [
                    c.id for c in all_chunks
                    if c.status == "active"
                    and c.layer != "gold"
                    and (include_immutable or not c.immutable)
                ]

            if not chunk_ids:
                return json.dumps({"status": "applied", "marked": 0})

            by_path: dict[str, list[str]] = defaultdict(list)
            for cid in chunk_ids:
                by_path[id_to_path.get(cid, args["path"])].append(cid)

            marked = 0
            for node_path, ids in by_path.items():
                op = StorageOperation(
                    operation="mark_stale",
                    target_path=node_path,
                    chunk_ids=ids,
                    reasoning_summary=args.get("reason", "Marked stale via MCP."),
                    force_immutable=include_immutable,
                )
                self._executor.apply(op)
                marked += len(ids)
            return json.dumps({"status": "applied", "marked": marked})

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
                            immutable=bool(item.get("immutable", False)),
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
            path = args["path"]
            ops: list[StorageOperation] = []
            ops.append(StorageOperation(
                operation="append_chunk",
                target_path=path,
                chunk=ChunkInput(
                    content=notes if notes else summary,
                    layer="bronze",
                    content_type="note",
                    source=self._client_source,
                ),
                reasoning_summary="Persisted session Bronze notes.",
            ))
            if notes or args.get("gold_aspect"):
                silver_chunk = ChunkInput(
                    content=summary,
                    layer="silver",
                    content_type="note",
                    source=self._client_source,
                )
                active_silver = [
                    c for c in self._store.get_chunks_by_path(path, include_children=False)  # type: ignore[attr-defined]
                    if c.layer == "silver" and c.status == "active"
                ]
                if active_silver:
                    # append_chunk(layer=silver) is rejected when a Silver already
                    # exists — rewrite the existing one instead.
                    ops.append(StorageOperation(
                        operation="update_silver",
                        target_path=path,
                        current_silver_id=active_silver[0].id,
                        chunk=silver_chunk,
                        reasoning_summary="Updated session Silver summary.",
                    ))
                else:
                    ops.append(StorageOperation(
                        operation="append_chunk",
                        target_path=path,
                        chunk=silver_chunk,
                        reasoning_summary="Persisted session Silver summary.",
                    ))
            batch = StorageOperationBatch(
                operations=ops,
                reasoning_summary="session_end: Bronze → Silver layering.",
            )
            batch_result = self._executor.apply_batch(batch)
            chunk_ids = [r.chunk_id for r in batch_result.results if r.chunk_id]
            out3: dict[str, Any] = {"status": batch_result.status, "chunk_ids": chunk_ids}
            if any(r.silver_too_large for r in batch_result.results):
                out3["silver_too_large"] = True
            if args.get("gold_aspect"):
                gold_op = StorageOperation(
                    operation="append_gold_aspect",
                    target_path=args["path"],
                    gold_aspect=args["gold_aspect"],
                    reasoning_summary="Updated Gold aspect via session_end.",
                )
                gold_result = self._executor.apply(gold_op)
                out3["gold_status"] = gold_result.status
                if gold_result.overflow_path:
                    out3["overflow_path"] = gold_result.overflow_path
                if gold_result.aspect_too_long:
                    out3["aspect_too_long"] = True
                if gold_result.gold_near_limit:
                    out3["gold_near_limit"] = True
            return json.dumps(out3)

        if name == "update_silver":
            path = args["path"]
            patch = args.get("patch")
            new_content = args.get("new_content")
            if (patch is None) == (new_content is None):
                raise ValueError("update_silver requires exactly one of 'new_content' or 'patch'")
            current_silver_id = args.get("current_silver_id")

            if patch is not None:
                if not patch:
                    raise ValueError("update_silver: 'patch' must contain at least one edit")
                active_silver = [
                    c for c in self._store.get_chunks_by_path(path, include_children=False)  # type: ignore[attr-defined]
                    if c.layer == "silver" and c.status == "active"
                ]
                if not active_silver:
                    raise ValueError(
                        f"update_silver(patch) rejected: no active Silver at '{path}'. "
                        f"Create the first Silver with append_chunk(layer=silver)."
                    )
                silver = active_silver[0]
                # OCC stays opt-in for patch mode: resolve the id if the caller
                # omitted it; honour it (and let the executor reject staleness)
                # if provided.
                if current_silver_id is None:
                    current_silver_id = silver.id
                content = silver.content
                for index, edit in enumerate(patch):
                    find = edit["find"]
                    occurrences = content.count(find)
                    if occurrences == 0:
                        raise ValueError(
                            f"update_silver patch[{index}]: 'find' text not found in the current Silver."
                        )
                    if occurrences > 1:
                        raise ValueError(
                            f"update_silver patch[{index}]: 'find' text occurs {occurrences} times — "
                            f"it must be unique. Include more surrounding context."
                        )
                    content = content.replace(find, edit["replace"], 1)
            else:
                content = new_content
                if current_silver_id is None:
                    raise ValueError("update_silver requires current_silver_id when using 'new_content'")

            op = StorageOperation(
                operation="update_silver",
                target_path=path,
                current_silver_id=current_silver_id,
                source_chunk_ids=args.get("source_chunk_ids") or [],
                chunk=ChunkInput(
                    content=content,
                    layer="silver",
                    content_type="note",
                    source=self._client_source,
                    confidence=args.get("confidence", 1.0),
                ),
                reasoning_summary=args.get("reasoning_summary") or "Updated Silver summary via MCP update_silver.",
            )
            result = self._executor.apply(op)
            out: dict[str, Any] = {"status": "applied", "chunk_id": result.chunk_id}
            if result.silver_too_large:
                out["silver_too_large"] = True
            return json.dumps(out)

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
            issues = Doctor(self._store, self._provider).run()  # type: ignore[arg-type]
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
                include_immutable=args.get("include_immutable", False),
                prune_empty_nodes=args.get("prune_empty_nodes", True),
                prune_vector_cache=args.get("prune_vector_cache", True),
                reclaim_space=args.get("reclaim_space", False),
            )
            return json.dumps(result, ensure_ascii=False, indent=2)

        if name == "ingest_file":
            return self._handle_ingest_file(args)

        if name == "get_service_chunk":
            return self._handle_get_service_chunk(args)

        if name == "get_service_chunks":
            return self._handle_get_service_chunks(args)

        if name == "mark_service_chunk":
            return self._handle_mark_service_chunk(args)

        if name == "batch_mark_service_chunks":
            return self._handle_batch_mark_service_chunks(args)

        if name == "finish_bronze_extraction":
            return self._handle_finish_bronze_extraction(args)

        if name == "submit_inventory_probes":
            return self._handle_submit_inventory_probes(args)

        if name == "complete_ingest":
            return self._handle_complete_ingest(args)

        if name == "ingest_url":
            return self._handle_ingest_url(args)

        raise KeyError(name)


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
