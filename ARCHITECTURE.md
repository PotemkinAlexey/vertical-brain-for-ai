# Vertical Brain — Architecture

Vertical Brain is a personal context warehouse / persistent memory system for
LLM agents. It stores knowledge in isolated vertical namespaces, accepts strict
storage operations, prevents cross-domain context leakage, and continuously
compacts raw notes into stable summaries.

## Core principles

1. **Namespace isolation.** Knowledge lives in slash-separated namespace paths
   (e.g. `WORK/DataArt/Databricks`). A read never crosses sibling branches
   unless an explicit horizontal link is followed.
2. **Map first, then lock.** Models orient with a namespace map, lock a target
   vertical, read a bounded context capsule, and only then expand links.
3. **Strict operations.** All writes go through validated `StorageOperation`
   objects — never ad-hoc mutation. Operations are schema-checked before any
   write touches storage.
4. **Layered knowledge.** Raw notes (Bronze) are compacted into canonical
   facts (Silver) and finally distilled into stable summaries (Gold).
5. **Search is for discovery, not context.** Search returns candidate handles.
   Model-facing content always comes from locked context capsules.

## Data model

### Chunk

A unit of stored content.

| Field | Meaning |
| --- | --- |
| `id` | UUID |
| `node_path` | Owning namespace path |
| `content` | Text content |
| `layer` | `bronze` / `silver` / `gold` |
| `content_type` | `fact`, `correction`, `decision`, `question`, `note`, `code`, `artifact` |
| `status` | `active`, `stale`, `superseded`, `legacy`, `contradicted`, `uncertain` |
| `source` | Origin marker (`manual`, `model`, optimizer source, etc.) |
| `confidence` | 0..1 |
| `lineage` | IDs of chunks this was derived from |
| `chunk_key` | Optional stable identity key (nullable) |
| `content_hash` | SHA-256 of whitespace-normalized content (computed) |
| `supersedes` | IDs of chunks this chunk replaces |
| `valid_from` / `valid_to` | ISO timestamps for temporal validity |
| `created_at` / `updated_at` | ISO timestamps |

`content_hash` is computed in `__post_init__` over normalized content, so two
chunks with the same text (modulo whitespace) always share a hash. This powers
duplicate detection in the optimizer and `vb doctor`.

### Node

A namespace in the tree: `id`, `path`, `name`, `parent_path`, `node_type`,
timestamps. Ancestor nodes are created automatically when a deep path is used.

### Link

A horizontal relationship between two namespaces: `source_path`,
`target_path`, `link_type`, `reason`. Symmetric link types (`peer`,
`related_to`) can be expanded from either side. Directional link types require
a `from_path` to disambiguate which endpoint is being expanded.

### Gold

Gold chunks hold distilled summaries. Content is backward-compatible:

- Plain text: `"Delta migration | AutoLoader streaming"`
- Structured JSON: `{"aspects": [...], "last_updated": "..."}`

`parse_gold_content()` handles both forms and returns an aspect list.

## Operation lifecycle

1. A model emits a `StorageOperation` or `StorageOperationBatch` (JSON).
2. The payload is checked against the JSON Schema in `model.json`.
3. `StorageOperationExecutor` runs semantic validation against current state.
4. `dry_run` reports validity without writing; `apply` / `apply_batch` write.
5. Batches run inside a transaction on backends that support it (SQLite).
6. Every successful applied operation is written to the operation audit log.

Operation types: `create_node`, `append_chunk`, `create_link`, `mark_stale`,
`supersede_chunk`, `append_gold_aspect`.

## Search-to-context lifecycle

```
MAP FIRST → LOCK TARGET VERTICAL → READ BOUNDED CONTEXT → EXPAND LINKS ON REQUEST → ANSWER
```

- `BrainSearch` / `EmbeddingSearch` produce ranked `SearchResult`s.
- `rank_paths_from_results()` aggregates per-path scores into a composite
  ranking (max score + average of top hits + a small hit-count bonus).
- `ContextSession` selects top-ranked paths and opens `LockedContext` capsules
  via `ContextLock`, bounded by a `ContextBudget`.
- Horizontal links are exposed as handles by default and expanded only when
  requested.

## Storage backend contract

Both backends implement the `StorageProvider` protocol
(`storage/protocol.py`): node/chunk/link CRUD, ancestor and peer lookups, and
`tree_text()`. `search()`, `gold_summary_path()`, and `transaction()` are
backend-specific and not part of the protocol. `log_audit()` is optional and
discovered via `hasattr`.

- **SQLiteStore** (default): transactional, FTS5 search index, automatic
  column migration for chunk versioning fields, `operation_audit` table.
- **JsonStore** (dev/debug): plain JSON files plus an append-only
  `operation_audit.jsonl`.

### SQLite concurrency model

SQLiteStore opens its connection with `PRAGMA journal_mode=WAL` and
`PRAGMA busy_timeout=5000`.  WAL mode allows one writer and multiple
concurrent readers without blocking each other.  Multiple `SQLiteStore`
instances pointing at the same database file are safe: each holds its own
connection, and the 5-second busy-timeout means a brief write contention
retries automatically rather than raising immediately.

**Do not share a single `SQLiteStore` instance across threads.**  The
connection is not thread-safe by default.  The right pattern for
multi-threaded code is one `SQLiteStore` per thread (or per request),
each opening its own connection to the same file.

For agent and MCP usage, prefer one `SQLiteStore` instance per
process/worker.  The MCP stdio server is single-process, so a single
instance is the right default.

## Known limitations / MVP status

- Routing uses a mock LLM unless a real provider or `--llm-response-file` is
  supplied; low-confidence routes ask for clarification rather than guessing.
- The optimizer is deterministic and intentionally simple: exact-duplicate
  stale marking plus namespace-bounded Silver compaction.
- Embedding search uses a mock provider by default; semantic quality depends
  on supplying a real embeddings endpoint.
- Gold structured content is supported on read; the executor still writes
  plain-text Gold.
- No UI — CLI and MCP stdio server only.
