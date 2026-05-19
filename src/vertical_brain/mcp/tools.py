"""MCP tool schemas (_TOOLS list) for Vertical Brain."""
from __future__ import annotations

from typing import Any

from vertical_brain.core.namespace_map import normalize_namespace_root_path

_ROOT_PATH_SCHEMA = {
    "type": "string",
    "description": (
        "Optional namespace prefix (e.g. PROJECTS/vertical-brain). "
        "Omit for the full tree. '/' means no filter — not a filesystem path."
    ),
}


def root_path_from_args(args: dict[str, Any]) -> str | None:
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
            "Add one or more short semantic routing tags to the Gold chunk of a "
            "namespace. Pass 'aspect' for a single tag or 'aspects' for several "
            "in one call. Returns overflow_paths if new sibling namespaces were created."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "aspect": {
                    "type": "string",
                    "description": "A single short search tag, ideally 30-100 characters.",
                },
                "aspects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Several short search tags to append in one call.",
                },
            },
            "required": ["path"],
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
            "Required count depends on depth (see session header)."
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
                "depth": {
                    "type": "string",
                    "enum": ["quick", "standard", "thorough"],
                    "description": "Extraction thoroughness. Same presets as ingest_file.",
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
