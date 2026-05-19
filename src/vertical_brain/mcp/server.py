"""MCP stdio server for Vertical Brain.

Protocol: JSON-RPC 2.0 over stdio, newline-delimited JSON (one object per line).
No external dependencies — pure stdlib.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import traceback
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vertical_brain.storage.protocol import StorageProvider

from vertical_brain.core.context_session import ContextSession
from vertical_brain.core.namespace_map import normalize_namespace_root_path
from vertical_brain.core.embedding_router import EmbeddingRouter
from vertical_brain.core.embedding_search import EmbeddingSearch
from vertical_brain.core.json_schema import format_json_schema_errors, validate_json_schema
from vertical_brain.core.models import Chunk, ChunkInput, LinkInput, StorageOperation, StorageOperationBatch
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.core.search import BrainSearch
from vertical_brain.llm.embedding import EmbeddingProvider, MockEmbeddingProvider
from vertical_brain.mcp.ingest_protocol import (
    DEFAULT_INGEST_DEPTH,
    DEFAULT_INGEST_MODE,
    INGEST_MODE_ROUTING,
    build_protocol_lines,
    calculate_coverage_score,
    inventory_probe_requirement,
    load_inventory_items,
    normalize_ingest_depth,
    normalize_ingest_mode,
    normalize_inventory_label,
    split_service_chunks,
    validate_inventory_probes,
    validate_namespace_ready_for_complete,
    validate_service_chunk_coverage,
    validate_skip_reason,
)

_PROTOCOL_VERSION = "2025-03-26"
_SERVER_VERSION = "0.1.0"


def _source_namespace(doc_slug: str) -> str:
    return f"SOURCES/{doc_slug}"


_ROOT_PATH_SCHEMA = {
    "type": "string",
    "description": (
        "Optional namespace prefix (e.g. PROJECTS/vertical-brain). "
        "Omit for the full tree. '/' means no filter — not a filesystem path."
    ),
}


def _root_path_from_args(args: dict[str, Any]) -> str | None:
    return normalize_namespace_root_path(args.get("root_path"))


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
                "root_path": _ROOT_PATH_SCHEMA,
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
                "root_path": _ROOT_PATH_SCHEMA,
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
                "root_path": _ROOT_PATH_SCHEMA,
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
                "root_path": _ROOT_PATH_SCHEMA,
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
                "root_path": _ROOT_PATH_SCHEMA,
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
                "root_path": _ROOT_PATH_SCHEMA,
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
                    "enum": ["fact", "reference", "correction", "decision", "question", "note", "code", "artifact"],
                },
                "source": {"type": "string", "description": "Who wrote this, e.g. 'model', 'user'"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "immutable": {
                    "type": "boolean",
                    "description": (
                        "Mark this Bronze chunk as immutable (default false). "
                        "Immutable chunks are reference artifacts (SWIFT schemas, specs, legal text) "
                        "that bypass Bronze dedup guards and are protected from mark_stale/supersede."
                    ),
                },
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
            "or omit to mark all active non-gold mutable chunks at path. "
            "Use recursive=true to include all descendant namespaces. "
            "Immutable chunks are skipped by default; set include_immutable=true "
            "to override (required for full namespace wipe before re-ingestion)."
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
                "recursive": {
                    "type": "boolean",
                    "description": "Also mark stale in all descendant namespaces (default false).",
                },
                "include_immutable": {
                    "type": "boolean",
                    "description": "Also mark immutable chunks stale (default false). Use only when wiping a namespace for re-ingestion.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "batch_append",
        "description": "Write multiple chunks in one call. All-or-nothing if using SQLite backend. Max 10 chunks per call — split larger sets into multiple batch_append calls.",
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
                            "immutable": {
                                "type": "boolean",
                                "description": "Mark this chunk as immutable (default false).",
                            },
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
                "include_immutable": {
                    "type": "boolean",
                    "description": "Also purge immutable inactive chunks. Requires force=true when applying.",
                },
                "prune_empty_nodes": {"type": "boolean", "description": "Default true"},
                "prune_vector_cache": {"type": "boolean", "description": "Default true"},
                "reclaim_space": {"type": "boolean", "description": "Run SQLite VACUUM after purging"},
            },
        },
    },
    {
        "name": "ingest_file",
        "description": (
            "Start a stateful ingest session. Prefer source_path for binary files (PDF, DOCX, DOC, HTML, ODT). "
            "Returns session_key, numbered service chunks, and an inline IRON RULES protocol. "
            "Default mode answer_complete: server blocks complete_ingest unless the namespace can "
            "answer questions without the source file. Follow every step until complete_ingest succeeds."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["answer_complete", "routing", "audit"],
                    "description": (
                        "answer_complete (default): enforce inventory, Bronze facts, Silver, "
                        "and coverage limits — source file will not be available later. "
                        "routing: discoverability-only ingest; Silver is marked not answer-complete. "
                        "audit: check an already-ingested namespace for coverage gaps — "
                        "no Bronze writing, just submit_inventory_probes then complete_ingest."
                    ),
                },
                "source_path": {
                    "type": "string",
                    "description": (
                        "Local path to the original source file. Required for binary formats "
                        "(.pdf, .docx, .doc, .html, .htm, .odt) so the server extracts complete "
                        "text and prevents summarized payloads."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Full text content of the attached plain-text file.",
                },
                "file_name": {
                    "type": "string",
                    "description": (
                        "Original file name, e.g. 'MT103.txt' or 'openapi.yaml'. "
                        "Defaults to the source_path basename when source_path is used."
                    ),
                },
                "expected_sha256": {
                    "type": "string",
                    "description": "Optional SHA-256 expected for the text payload; mismatches are rejected.",
                },
                "expected_size_bytes": {
                    "type": "integer",
                    "description": "Optional UTF-8 byte count expected for the text payload; mismatches are rejected.",
                },
                "authority": {
                    "type": "string",
                    "description": (
                        "Issuing authority or origin of the document "
                        "(e.g. 'SWIFT', 'ISO', 'Internal', 'Vendor'). "
                        "Stored as source metadata; the namespace is SOURCES/{slug}."
                    ),
                },
                "doc_slug": {
                    "type": "string",
                    "description": (
                        "Short identifier for the document used in the namespace path "
                        "(e.g. 'MT103', 'openapi-v2'). "
                        "Defaults to the file name without extension."
                    ),
                },
                "depth": {
                    "type": "string",
                    "enum": ["quick", "standard", "thorough"],
                    "description": (
                        "Extraction thoroughness. quick: 50% max skip, 60% coverage (fast scan). "
                        "standard (default): 25% max skip, 90% coverage. "
                        "thorough: 10% max skip, 95% coverage, min 5 inventory probes. "
                        "If not specified, ask the user before starting."
                    ),
                },
                "force": {
                    "type": "boolean",
                    "description": (
                        "Set to true to re-ingest a file that was already ingested "
                        "(same sha256 found in registry). Default false — duplicate is rejected."
                    ),
                },
            },
        },
    },
    {
        "name": "get_service_chunk",
        "description": "Read the full content of one service chunk from an active ingest session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_key": {"type": "string", "description": "Session key returned by ingest_file."},
                "chunk_id": {"type": "string", "description": "Chunk ID from the ingest_file chunk list."},
            },
            "required": ["session_key", "chunk_id"],
        },
    },
    {
        "name": "get_service_chunks",
        "description": (
            "Read up to 10 service chunks in one call (batch). "
            "Pass chunk_ids from the ingest_file list, or start_index/count for a range."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_key": {"type": "string"},
                "chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific chunk IDs to fetch (max 10).",
                },
                "start_index": {
                    "type": "integer",
                    "description": "0-based index into the session chunk list (use with count).",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of chunks from start_index (default 5, max 10).",
                },
            },
            "required": ["session_key"],
        },
    },
    {
        "name": "mark_service_chunk",
        "description": (
            "Mark a service chunk as processed. "
            "Call with status='extracted' after writing Bronze facts, "
            "or status='skipped' only when the chunk is boilerplate, empty/formatting noise, "
            "irrelevant to the source purpose, or a duplicate of Bronze already written."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_key": {"type": "string"},
                "chunk_id": {"type": "string"},
                "status": {"type": "string", "enum": ["extracted", "skipped"]},
                "skip_reason": {"type": "string", "description": "Required when status='skipped'."},
            },
            "required": ["session_key", "chunk_id", "status"],
        },
    },
    {
        "name": "batch_mark_service_chunks",
        "description": (
            "Mark multiple service chunks as extracted or skipped in a single call. "
            "Use after writing Bronze for a whole section to avoid per-chunk round-trips. "
            "Max 50 marks per call."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_key": {"type": "string"},
                "marks": {
                    "type": "array",
                    "description": "Array of {chunk_id, status, skip_reason?} objects.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chunk_id": {"type": "string"},
                            "status": {"type": "string", "enum": ["extracted", "skipped"]},
                            "skip_reason": {"type": "string"},
                        },
                        "required": ["chunk_id", "status"],
                    },
                    "maxItems": 50,
                },
            },
            "required": ["session_key", "marks"],
        },
    },
    {
        "name": "finish_bronze_extraction",
        "description": (
            "Validate that all service chunks have been processed (extracted or skipped). "
            "Returns an error listing pending chunk IDs if any remain. "
            "Call this before writing Silver."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_key": {"type": "string"},
            },
            "required": ["session_key"],
        },
    },
    {
        "name": "submit_inventory_probes",
        "description": (
            "Submit spot-check probes before complete_ingest (answer_complete mode). "
            "Each probe names an [INVENTORY] item and cites active Bronze chunk_id(s) that answer it. "
            "Required count: max(3, 30% of inventory lines)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_key": {"type": "string"},
                "probes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item": {"type": "string"},
                            "chunk_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["item", "chunk_ids"],
                    },
                },
            },
            "required": ["session_key", "probes"],
        },
    },
    {
        "name": "complete_ingest",
        "description": (
            "Complete the ingest session after Silver has been written. "
            "Requires finish_bronze_extraction and (answer_complete mode) artifact, inventory, "
            "Bronze facts, and Silver in storage. Cleans up service chunks from memory."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_key": {"type": "string"},
            },
            "required": ["session_key"],
        },
    },
    {
        "name": "ingest_url",
        "description": (
            "Fetch a URL and start the same stateful ingest session as ingest_file. "
            "Supports plain text, Markdown, JSON, YAML, and HTML (tags stripped). "
            "Follow the inline IRON RULES protocol until complete_ingest succeeds."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch (http or https).",
                },
                "mode": {
                    "type": "string",
                    "enum": ["answer_complete", "routing", "audit"],
                    "description": "Default answer_complete. Same semantics as ingest_file.",
                },
                "authority": {
                    "type": "string",
                    "description": (
                        "Issuing authority or origin (e.g. 'SWIFT', 'ISO', 'stripe.com'). "
                        "Stored as source metadata. If omitted, derived from the URL hostname."
                    ),
                },
                "doc_slug": {
                    "type": "string",
                    "description": (
                        "Short identifier for the document (e.g. 'payment-intents-api'). "
                        "Defaults to the last path segment of the URL."
                    ),
                },
                "force": {
                    "type": "boolean",
                    "description": "Bypass duplicate content_hash check in ingest_registry.",
                },
            },
            "required": ["url"],
        },
    },
]

_TOOLS_BY_NAME: dict[str, dict[str, Any]] = {tool["name"]: tool for tool in _TOOLS}


from vertical_brain.mcp.extractors.html import strip_html as _strip_html


from vertical_brain.mcp.extractors import extract_source_text as _extract_source_text


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
        self._ingest_sessions: dict[str, dict[str, Any]] = {}
        self._load_ingest_sessions_from_store()

    def _load_ingest_sessions_from_store(self) -> None:
        list_sessions = getattr(self._store, "list_ingest_sessions", None)
        if not callable(list_sessions):
            return
        for row in list_sessions():
            try:
                session = json.loads(row["session_json"])
            except json.JSONDecodeError:
                continue
            key = session.get("session_key") or row["session_key"]
            self._ingest_sessions[key] = session

    def _persist_ingest_session(self, session: dict[str, Any]) -> None:
        save = getattr(self._store, "save_ingest_session", None)
        if not callable(save):
            return
        save(
            session["session_key"],
            session.get("state", "EXTRACTING_BRONZE"),
            json.dumps(session, ensure_ascii=False),
        )

    def _delete_persisted_ingest_session(self, session_key: str) -> None:
        delete = getattr(self._store, "delete_ingest_session", None)
        if callable(delete):
            delete(session_key)

    def _cancel_ingest_sessions_for_namespace(self, source_namespace: str) -> int:
        """Remove in-memory and persisted ingest sessions targeting *source_namespace*."""
        keys_to_remove: set[str] = set()
        for key, session in self._ingest_sessions.items():
            if session.get("source_namespace") == source_namespace:
                keys_to_remove.add(key)

        list_sessions = getattr(self._store, "list_ingest_sessions", None)
        if callable(list_sessions):
            for row in list_sessions():
                key = row.get("session_key") or ""
                if not key:
                    continue
                try:
                    session = json.loads(row["session_json"])
                except json.JSONDecodeError:
                    keys_to_remove.add(key)
                    continue
                if session.get("source_namespace") == source_namespace:
                    keys_to_remove.add(key)

        for key in keys_to_remove:
            self._ingest_sessions.pop(key, None)
            self._delete_persisted_ingest_session(key)
        return len(keys_to_remove)

    def _require_ingest_session(self, session_key: str) -> dict[str, Any]:
        session = self._ingest_sessions.get(session_key)
        if session is not None:
            return session
        load = getattr(self._store, "get_ingest_session", None)
        if callable(load):
            row = load(session_key)
            if row is not None:
                session = json.loads(row["session_json"])
                self._ingest_sessions[session_key] = session
                return session
        raise ValueError(f"No active ingest session: {session_key!r}")

    def _wipe_source_namespace(self, path: str) -> int:
        from collections import defaultdict

        get_chunks = getattr(self._store, "get_chunks_by_path", None)
        if not callable(get_chunks):
            return 0
        all_chunks = get_chunks(path, include_children=True)
        chunk_ids = [
            c.id for c in all_chunks
            if c.status == "active" and c.layer != "gold"
        ]
        if not chunk_ids:
            return 0
        id_to_path = {c.id: c.node_path for c in all_chunks}
        by_path: dict[str, list[str]] = defaultdict(list)
        for cid in chunk_ids:
            by_path[id_to_path[cid]].append(cid)
        marked = 0
        for node_path, ids in by_path.items():
            op = StorageOperation(
                operation="mark_stale",
                target_path=node_path,
                chunk_ids=ids,
                reasoning_summary="Auto-wipe before force re-ingest.",
                force_immutable=True,
            )
            self._executor.apply(op)
            marked += len(ids)
        return marked

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
                root_path=_root_path_from_args(args),
                max_depth=args.get("max_depth"),
                summary_max_chars=args.get("summary_chars", 200),
            )

        if name == "namespace_map":
            return self._session.namespace_map(
                root_path=_root_path_from_args(args),
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
                root_path=_root_path_from_args(args),
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
                root_path=_root_path_from_args(args),
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
                root_path=_root_path_from_args(args),
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
                root_path=_root_path_from_args(args),
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
            text = json.dumps(out)
            layer = args.get("layer", "bronze")
            if layer == "bronze":
                text += (
                    "\n\n---\nAGENTS: Bronze written → update Silver now (update_silver). "
                    "Write after every decision or code change — not only at session_end."
                )
            return text

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
            from collections import defaultdict
            include_children = bool(args.get("recursive", False))
            include_immutable = bool(args.get("include_immutable", False))
            explicit_ids: list[str] = args.get("chunk_ids") or []

            if explicit_ids:
                # Explicit IDs: trust the caller; group by actual node_path.
                all_chunks = self._store.get_chunks_by_path(  # type: ignore[attr-defined]
                    args["path"], include_children=True
                )
                id_to_path = {c.id: c.node_path for c in all_chunks}
                chunk_ids = explicit_ids
            else:
                # Auto-collect: active non-gold chunks, optionally recursive + immutable.
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

            # Group by node_path — mark_stale op requires target_path to match chunk's namespace.
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

    def _start_ingest_session(
        self,
        *,
        content: str,
        file_name: str,
        authority: str,
        doc_slug: str,
        ingest_mode: str,
        ingest_depth: str,
        force: bool,
        source_hash: str | None = None,
        source_size: int | None = None,
    ) -> str:
        import secrets

        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        file_size = len(content.encode("utf-8"))
        source_ns = _source_namespace(doc_slug)

        find_ingest = getattr(self._store, "find_ingest_by_hash", None)
        if callable(find_ingest) and not force:
            existing = find_ingest(file_hash)
            if existing:
                raise ValueError(
                    f"This file was already ingested.\n"
                    f"  file: {existing['file_name']}\n"
                    f"  namespace: {existing['source_namespace']}\n"
                    f"  ingested_at: {existing['ingested_at']}\n"
                    f"  sha256: {file_hash}\n\n"
                    f"Pass force=true to re-ingest (wipe namespace first if replacing content)."
                )

        wiped = 0
        cancelled_sessions = 0
        if force:
            wiped = self._wipe_source_namespace(source_ns)
            cancelled_sessions = self._cancel_ingest_sessions_for_namespace(source_ns)

        session_key = secrets.token_hex(8)
        service_chunks = split_service_chunks(content, session_key)
        session = {
            "session_key": session_key,
            "file_name": file_name,
            "authority": authority,
            "doc_slug": doc_slug,
            "source_namespace": source_ns,
            "content_hash": file_hash,
            "content_size": file_size,
            "source_hash": source_hash,
            "source_size": source_size,
            "ingest_mode": ingest_mode,
            "ingest_depth": ingest_depth,
            "state": "EXTRACTING_BRONZE",
            "chunks": service_chunks,
            "inventory_probes": [],
            "force_wiped": wiped,
        }
        self._ingest_sessions[session_key] = session
        self._persist_ingest_session(session)
        lines = build_protocol_lines(session)
        if force and wiped:
            lines += [
                "",
                f"force re-ingest: marked {wiped} prior chunk(s) stale under {source_ns}.",
            ]
        if cancelled_sessions > 0:
            lines += [
                "",
                f"cancelled_sessions: {cancelled_sessions}",
            ]
        return "\n".join(lines)

    def _handle_ingest_file(self, args: dict[str, Any]) -> str:
        source_path = (args.get("source_path") or "").strip()
        source_hash: str | None = None
        source_size: int | None = None

        if source_path:
            content, source_hash, source_size = _extract_source_text(source_path)
            file_name = args.get("file_name") or os.path.basename(source_path)
        else:
            if "content" not in args:
                raise ValueError("ingest_file requires either source_path or content")
            if "file_name" not in args:
                raise ValueError("ingest_file requires file_name when content is supplied")
            content = args["content"]
            file_name = args["file_name"]
            ext = os.path.splitext(file_name)[1].lower()
            if ext in {".pdf", ".docx", ".doc", ".html", ".htm", ".odt"}:
                raise ValueError(
                    f"{ext} ingestion must use source_path so the server extracts the complete text. "
                    "Passing caller-supplied content for binary formats is rejected to prevent summarized payloads."
                )

        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        file_size = len(content.encode("utf-8"))
        expected_hash = (args.get("expected_sha256") or "").strip()
        if expected_hash and expected_hash != file_hash:
            raise ValueError(
                f"ingest_file integrity check failed: expected_sha256={expected_hash}, "
                f"actual_sha256={file_hash}"
            )
        expected_size = args.get("expected_size_bytes")
        if expected_size is not None and int(expected_size) != file_size:
            raise ValueError(
                f"ingest_file integrity check failed: expected_size_bytes={expected_size}, "
                f"actual_size_bytes={file_size}"
            )

        authority: str = (args.get("authority") or "").strip().replace(" ", "_")
        raw_slug = args.get("doc_slug") or os.path.splitext(file_name)[0]
        doc_slug = raw_slug.strip().replace(" ", "_").replace(".", "_")
        ingest_mode = normalize_ingest_mode(args.get("mode"))
        ingest_depth = normalize_ingest_depth(args.get("depth"))

        return self._start_ingest_session(
            content=content,
            file_name=file_name,
            authority=authority,
            doc_slug=doc_slug,
            ingest_mode=ingest_mode,
            ingest_depth=ingest_depth,
            force=bool(args.get("force", False)),
            source_hash=source_hash,
            source_size=source_size,
        )

    def _handle_get_service_chunk(self, args: dict[str, Any]) -> str:
        session_key = args.get("session_key", "")
        chunk_id = args.get("chunk_id", "")
        session = self._require_ingest_session(session_key)
        chunk = next((c for c in session["chunks"] if c["id"] == chunk_id), None)
        if chunk is None:
            raise ValueError(f"Chunk {chunk_id!r} not found in session {session_key!r}")
        return json.dumps({
            "chunk_id": chunk_id,
            "index": chunk["index"],
            "chunk_type": chunk["chunk_type"],
            "status": chunk["status"],
            "content": chunk["content"],
        }, ensure_ascii=False)

    def _ingest_step_hint(self, session: dict[str, Any]) -> str:
        """Return a short step label + progress string for the current ingest phase."""
        total = len(session.get("chunks") or [])
        processed = sum(1 for c in (session.get("chunks") or []) if c["status"] != "pending")
        pending = total - processed
        state = session.get("state", "")
        ns = session.get("source_namespace", "")
        if state == "BRONZE_COMPLETE":
            return "STEP 5 ✓ — Bronze complete. Next: STEP 6 — write Silver per sub-namespace, then submit_inventory_probes, complete_ingest."
        pct = int(processed / total * 100) if total else 0
        return (
            f"STEP 4 — Bronze extraction | {ns} | "
            f"{processed}/{total} chunks processed ({pct}%) | {pending} pending"
        )

    def _handle_get_service_chunks(self, args: dict[str, Any]) -> str:
        session_key = args.get("session_key", "")
        session = self._require_ingest_session(session_key)

        chunk_ids = args.get("chunk_ids") or []
        if chunk_ids:
            selected = []
            for chunk_id in chunk_ids[:10]:
                chunk = next((c for c in session["chunks"] if c["id"] == chunk_id), None)
                if chunk is None:
                    raise ValueError(f"Chunk {chunk_id!r} not found in session {session_key!r}")
                selected.append(chunk)
        else:
            start = int(args.get("start_index", 0))
            count = min(int(args.get("count", 5)), 10)
            if start < 0:
                raise ValueError("start_index must be >= 0")
            selected = session["chunks"][start : start + count]

        return json.dumps(
            {
                "session_key": session_key,
                "step": self._ingest_step_hint(session),
                "chunks": [
                    {
                        "chunk_id": c["id"],
                        "index": c["index"],
                        "chunk_type": c["chunk_type"],
                        "status": c["status"],
                        "content": c["content"],
                    }
                    for c in selected
                ],
            },
            ensure_ascii=False,
        )

    def _handle_mark_service_chunk(self, args: dict[str, Any]) -> str:
        session_key = args.get("session_key", "")
        chunk_id = args.get("chunk_id", "")
        status = args.get("status", "")
        skip_reason = args.get("skip_reason")
        if status not in ("extracted", "skipped"):
            raise ValueError(f"status must be 'extracted' or 'skipped', got {status!r}")
        if status == "skipped" and not skip_reason:
            raise ValueError("skip_reason is required when status='skipped'")
        session = self._require_ingest_session(session_key)
        if status == "skipped":
            validate_skip_reason(str(skip_reason), mode=session.get("ingest_mode", DEFAULT_INGEST_MODE))
        chunk = next((c for c in session["chunks"] if c["id"] == chunk_id), None)
        if chunk is None:
            raise ValueError(f"Chunk {chunk_id!r} not found in session {session_key!r}")
        chunk["status"] = status
        chunk["skip_reason"] = skip_reason
        self._persist_ingest_session(session)
        pending = sum(1 for c in session["chunks"] if c["status"] == "pending")
        return json.dumps(
            {"ok": True, "chunk_id": chunk_id, "status": status, "remaining_pending": pending},
            ensure_ascii=False,
        )

    def _handle_batch_mark_service_chunks(self, args: dict[str, Any]) -> str:
        session_key = args.get("session_key", "")
        marks = args.get("marks") or []
        if not marks:
            raise ValueError("marks must be a non-empty array")
        if len(marks) > 50:
            raise ValueError(f"batch_mark_service_chunks: max 50 marks per call, got {len(marks)}")
        session = self._require_ingest_session(session_key)
        ingest_mode = session.get("ingest_mode", DEFAULT_INGEST_MODE)
        chunk_index = {c["id"]: c for c in session["chunks"]}
        results = []
        for mark in marks:
            chunk_id = mark.get("chunk_id", "")
            status = mark.get("status", "")
            skip_reason = mark.get("skip_reason")
            if status not in ("extracted", "skipped"):
                raise ValueError(f"status must be 'extracted' or 'skipped', got {status!r} for chunk {chunk_id!r}")
            if status == "skipped" and not skip_reason:
                raise ValueError(f"skip_reason is required when status='skipped' (chunk {chunk_id!r})")
            if status == "skipped":
                validate_skip_reason(str(skip_reason), mode=ingest_mode)
            chunk = chunk_index.get(chunk_id)
            if chunk is None:
                raise ValueError(f"Chunk {chunk_id!r} not found in session {session_key!r}")
            chunk["status"] = status
            chunk["skip_reason"] = skip_reason
            results.append({"chunk_id": chunk_id, "status": status})
        self._persist_ingest_session(session)
        pending = sum(1 for c in session["chunks"] if c["status"] == "pending")
        return json.dumps(
            {
                "ok": True,
                "marked": len(results),
                "remaining_pending": pending,
                "step": self._ingest_step_hint(session),
                "results": results,
            },
            ensure_ascii=False,
        )

    def _handle_finish_bronze_extraction(self, args: dict[str, Any]) -> str:
        session_key = args.get("session_key", "")
        session = self._require_ingest_session(session_key)
        pending = [c for c in session["chunks"] if c["status"] == "pending"]
        if pending:
            ids = [c["id"] for c in pending[:5]]
            more = f" ... (+{len(pending) - 5} more)" if len(pending) > 5 else ""
            raise ValueError(
                f"{len(pending)} chunk(s) still pending — mark them extracted or skipped first. "
                f"Pending: {ids}{more}"
            )
        validate_service_chunk_coverage(session, phase="finish_bronze_extraction")
        extracted = sum(1 for c in session["chunks"] if c["status"] == "extracted")
        skipped = sum(1 for c in session["chunks"] if c["status"] == "skipped")
        session["state"] = "BRONZE_COMPLETE"
        self._persist_ingest_session(session)
        return json.dumps({
            "status": "ok",
            "step": "STEP 5 ✓ — finish_bronze_extraction passed.",
            "extracted": extracted,
            "skipped": skipped,
            "total": len(session["chunks"]),
            "next_action": (
                "STEP 6 — write Silver per sub-namespace as [chunk_id] one-liner index, "
                f"then root Silver at {session['source_namespace']}. "
                "STEP 6b — submit_inventory_probes. STEP 7 — complete_ingest."
            ),
        }, ensure_ascii=False)

    def _handle_submit_inventory_probes(self, args: dict[str, Any]) -> str:
        session_key = args.get("session_key", "")
        probes = args.get("probes") or []
        if not probes:
            raise ValueError("probes must be a non-empty list of {item, chunk_ids} objects")
        session = self._require_ingest_session(session_key)
        existing: list[dict] = list(session.get("inventory_probes") or [])
        existing_by_item: dict[str, dict] = {
            normalize_inventory_label(p["item"]): p for p in existing if p.get("item")
        }
        for probe in probes:
            item = probe.get("item", "")
            key = normalize_inventory_label(item)
            if key:
                existing_by_item[key] = probe
        session["inventory_probes"] = list(existing_by_item.values())
        self._persist_ingest_session(session)

        inventory_items = load_inventory_items(self._store, session["source_namespace"])
        inventory_count = len(inventory_items)
        required = inventory_probe_requirement(inventory_count)

        total_accumulated = len(session["inventory_probes"])
        return json.dumps({
            "status": "ok",
            "submitted": total_accumulated,
            "required_for_complete": required,
            "inventory_item_count": inventory_count,
        }, ensure_ascii=False)

    def _handle_complete_ingest(self, args: dict[str, Any]) -> str:
        session_key = args.get("session_key", "")
        session = self._require_ingest_session(session_key)
        if session["state"] != "BRONZE_COMPLETE":
            raise ValueError(
                f"Cannot complete ingest: state is {session['state']!r}. "
                "Call finish_bronze_extraction first."
            )
        validate_service_chunk_coverage(session, phase="complete_ingest")
        storage_check = validate_namespace_ready_for_complete(self._store, session)
        probe_check = validate_inventory_probes(session, self._store)
        coverage = calculate_coverage_score(session, self._store)
        extracted = sum(1 for c in session["chunks"] if c["status"] == "extracted")
        skipped = sum(1 for c in session["chunks"] if c["status"] == "skipped")
        source_ns = session["source_namespace"]
        ingest_mode = session.get("ingest_mode", DEFAULT_INGEST_MODE)

        # Routing mode: write an explicit honesty note in Bronze so storage reflects the limitation.
        if ingest_mode == INGEST_MODE_ROUTING:
            self._store.save_chunk(Chunk(
                node_path=source_ns,
                layer="bronze",
                content_type="note",
                content=(
                    "[ROUTING ONLY] Discoverability ingest — not answer-complete. "
                    "Silver contains section index only. "
                    "Do not cite this namespace for authoritative answers."
                ),
            ))

        register_ingest = getattr(self._store, "register_ingest", None)
        if callable(register_ingest):
            register_ingest(
                session["content_hash"],
                source_ns,
                session["file_name"],
            )

        del self._ingest_sessions[session_key]
        self._delete_persisted_ingest_session(session_key)
        return json.dumps({
            "status": "completed",
            "source_namespace": source_ns,
            "ingest_mode": ingest_mode,
            "bronze_extracted": extracted,
            "service_chunks_skipped": skipped,
            "storage_check": storage_check,
            "inventory_probes": probe_check,
            "coverage_score": coverage,
            "session_key": session_key,
            "next_steps": (
                "MANDATORY: call append_gold_aspect for the source namespace AND each "
                "sub-namespace to add short routing tags (30-100 chars each). "
                "Gold is how future agents discover this content via session_start. "
                "Without Gold, the document is invisible to routing."
            ),
        }, ensure_ascii=False)

    def _handle_ingest_url(self, args: dict[str, Any]) -> str:
        import urllib.parse
        import urllib.request

        url: str = args["url"]
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Only http/https URLs are supported, got: {url!r}")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vertical-brain/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_bytes: bytes = resp.read()
                content_type: str = resp.headers.get_content_type() or ""
        except urllib.error.URLError as exc:
            raise ValueError(f"Failed to fetch URL: {exc}") from exc

        charset = "utf-8"
        if hasattr(resp.headers, "get_content_charset"):
            charset = resp.headers.get_content_charset() or "utf-8"
        try:
            raw_text = raw_bytes.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            raw_text = raw_bytes.decode("utf-8", errors="replace")

        if "html" in content_type:
            content = _strip_html(raw_text)
        else:
            content = raw_text

        parsed = urllib.parse.urlparse(url)
        authority: str = (args.get("authority") or parsed.hostname or "").strip().replace(" ", "_")
        path_parts = [p for p in parsed.path.split("/") if p]
        default_slug = path_parts[-1] if path_parts else parsed.hostname or "doc"
        raw_slug = args.get("doc_slug") or default_slug
        doc_slug = raw_slug.strip().replace(" ", "_").replace(".", "_")
        file_name = doc_slug or "url_document"
        ingest_mode = normalize_ingest_mode(args.get("mode"))
        ingest_depth = normalize_ingest_depth(args.get("depth"))

        return self._start_ingest_session(
            content=content,
            file_name=file_name,
            authority=authority,
            doc_slug=doc_slug,
            ingest_mode=ingest_mode,
            ingest_depth=ingest_depth,
            force=bool(args.get("force", False)),
        )

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
