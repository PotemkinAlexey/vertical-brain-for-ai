# Vertical Brain — Claude Code Guide

## What This Project Is

A personal context lakehouse for AI assistants. Knowledge is stored in hierarchical namespaces (`WORK/DataArt/Databricks`), organized into Bronze → Silver → Gold quality layers, and served to models through locked context capsules with strict isolation.

**Zero Python runtime dependencies.** Core + storage + MCP server use only the Python standard library. The HTTP embedding provider uses `urllib` from stdlib. Optional semantic search requires an external OpenAI-compatible embedding endpoint.

---

## Repository Layout

```
src/vertical_brain/
├── core/
│   ├── models.py             — All data classes (Chunk, Node, Link, StorageOperation, …)
│   ├── operations.py         — StorageOperationExecutor: validate + apply + audit + Bronze dedup guard
│   ├── optimizer.py          — SimpleOptimizer: dedup + Silver compaction + decay + cross-namespace link discovery
│   ├── gold.py               — GoldAspect v2, parse/serialize, legacy GoldDocument
│   ├── search.py             — BrainSearch: lexical FTS + path ranking
│   ├── embedding_search.py   — EmbeddingSearch: cosine + persistent vector cache
│   ├── embedding_router.py   — EmbeddingRouter: route by embedding similarity
│   ├── context_lock.py       — ContextLock: builds locked context capsules
│   ├── context_session.py    — ContextSession: map + search + lock orchestration
│   ├── namespace_map.py      — NamespaceMapBuilder: model-facing orientation map
│   ├── router.py             — LLMRouter + StorageModel: validate route decisions
│   ├── doctor.py             — Doctor: storage integrity checks
│   └── json_schema.py        — Stdlib JSON Schema validator (no external deps)
│
├── storage/
│   ├── protocol.py           — StorageProvider Protocol (runtime-checkable); includes has_active_chunk_with_hash
│   ├── sqlite_store.py       — SQLiteStore: WAL, FTS5, transactions, vector cache, composite dedup index
│   ├── json_store.py         — JsonStore: human-readable files, dev/debug
│   └── thread_local_store.py — ThreadLocalSQLiteStoreProxy
│
├── llm/
│   ├── embedding.py          — EmbeddingProvider Protocol, Mock, HttpEmbeddingProvider
│   └── mock_llm.py           — MockLLM for testing
│
├── mcp/
│   ├── server.py             — MCP stdio server (JSON-RPC 2.0, NDJSON over stdio)
│   └── ingest_protocol.py    — Ingest IRON RULES, splitting, answer_complete gates
│
└── cli/
    └── main.py               — `vb` CLI entry point
```

Test files mirror `src/` — there is one test file per module, plus cross-cutting tests (`test_occ.py`, `test_version_propagation.py`, etc.).

---

## Running Tests

```bash
pytest                                    # full suite (499 tests)
pytest tests/test_operations.py -v       # single file
pytest -k "occ or version" -v            # keyword filter
pytest --tb=short 2>&1 | tail -20        # summary view
```

All tests use `tmp_path` fixtures — no shared state, no cleanup required.

---

## Key Invariants (Do Not Break)

These are tested explicitly in `test_occ.py`, `test_version_propagation.py`, `test_rename_namespace.py`, and others:

1. **OCC check is inside the transaction.** In `StorageOperationExecutor.apply_batch`, the `branch_path`/`start_version` check must be inside `with context:`, not before it.

2. **`valid_to` is set automatically** when a chunk transitions to `stale`, `superseded`, `legacy`, or `contradicted`. See `_update_chunk_status` in `operations.py`.

3. **`node.version` increments and `node.is_dirty` is set** on every chunk write, via `_mark_ancestors_dirty`. This propagates up the path — writing to `WORK/A/B` increments versions of `WORK/A/B`, `WORK/A`, and `WORK`.

4. **Gold aspects are upserted by text.** In `_apply_append_gold_aspect`, exact-text duplicates refresh `updated_at` without creating a new entry.

5. **Max 20 Gold aspects per node.** Overflow creates a sibling namespace `{path}_2`.

6. **`rename_namespace` is atomic in SQLite.** It must be wrapped in `with self.transaction():`.

7. **The optimizer performs a bounded snapshot read** via `_snapshot_branch`: branch chunks once, node version once, and links once only when decay is enabled. Planning (`_build_plan`) is pure over that snapshot — no further I/O. Changes are applied as a single transactional batch.

8. **Bronze dedup is enforced at the executor level, not the store level.** `_apply_validated` checks exact-hash duplicates via `store.has_active_chunk_with_hash()` (hard block) and near-duplicates via `_find_similar_bronze()` (soft warning in `OperationResult.similar_bronze`). The check applies only to `layer="bronze"` — Silver and Gold are exempt. The composite index `idx_chunks_dedup (node_path, status, content_hash)` makes the hard block O(log n).

9. **`append_gold_aspect` requires an active Silver chunk at the same namespace.** The executor rejects the write with a descriptive `ValueError` if no active Silver exists. Gold must be grounded in Silver — write Silver first via `update_silver`, then promote to Gold.

---

## Adding a New Operation

1. Add the operation name to the `operation` enum in `models.py` (`StorageOperation.operation` type hint).
2. Add a handler `_apply_{name}` in `StorageOperationExecutor` in `operations.py`.
3. Add routing in `apply` / `apply_batch` dispatch.
4. Update the JSON Schema in `data/namespaces/model.json` (the `operation` enum).
5. Write a test in `tests/test_operations.py`.

---

## Adding a New Storage Backend

Implement `StorageProvider` from `storage/protocol.py` — it is a `runtime_checkable` Protocol, no base class needed. Optionally implement:
- `VectorCacheStorageProvider` for embedding cache
- `EmbeddingSchemaStorageProvider` for model name/dimension tracking

`EmbeddingSearch` and `EmbeddingRouter` detect these via `getattr` and degrade gracefully without them.

---

## Gold Aspects

Gold content is stored as Gold-layer chunks attached to a node. The canonical format (v2 JSON) is:

```json
{"aspects": [{"id": "...", "text": "...", "updated_at": "..."}]}
```

**Always use `append_gold_aspect` (MCP) or the `append_gold_aspect` operation (StorageOperationBatch) to write Gold.** Never mutate Gold chunks directly or construct the JSON by hand. `GoldBuilder` / `LlmGoldBuilder` were removed — batch distillation is not wired.

Use `parse_gold_aspects(content)` → `list[GoldAspect]` to read, and `serialize_gold_aspects(aspects)` → `str` to write internally. `parse_gold_content(content)` → `list[str]` extracts just the text strings (backward compatible with plain text, v1 JSON, and legacy `GoldDocument` JSON with a `facts` array).

---

## Embedding Vector Cache

Vectors are keyed by `(content_hash, model_name)`. The model name is recorded in the store schema (`set_embedding_schema`). If you change the embedding model, call `trigger_reindexing(new_provider)` to purge old vectors and re-embed.

`EmbeddingSearch.__init__` raises `IncompatibleEmbeddingModelError` if the provider's model name doesn't match what's stored — this is intentional and should not be silenced.

---

## SQLite Concurrency

- WAL mode: N concurrent readers + 1 writer
- One `SQLiteStore` per thread. Do not share instances across threads.
- For multi-threaded code, use `ThreadLocalSQLiteStoreProxy` — it creates one `SQLiteStore` per thread lazily.

---

## Commit Style

This repo uses conventional commit messages. Keep messages short and factual. Multi-line body when the change is non-trivial. No trailing "Co-authored" lines needed unless explicitly requested.

---

## Things to Avoid

- **Do not add external Python dependencies.** The zero-runtime-dependency constraint is intentional.
- **Do not bypass the JSON Schema validation.** Never call `_apply_*` methods directly from outside the executor.
- **Do not write raw strings to Gold.** Always use `append_gold_aspect` / `serialize_gold_aspects` — never mutate Gold chunks directly.
- **Do not share `SQLiteStore` across threads.** Use `ThreadLocalSQLiteStoreProxy`.
- **Do not read storage inside `_build_plan`.** The optimizer is pure over its snapshot input.
- **For agent-facing usage rules** (optimize, vacuum, rename_namespace, write layering) — see `AGENTS.md`.
