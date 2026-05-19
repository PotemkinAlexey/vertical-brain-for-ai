# Vertical Brain — Claude Code Guide

## What This Project Is

A personal context lakehouse for AI assistants. Knowledge is stored in hierarchical namespaces (`WORK/DataArt/Databricks`), organized into Bronze → Silver → Gold quality layers, and served to models through locked context capsules with strict isolation.

**Zero Python runtime dependencies.** Core + storage + MCP server use only the Python standard library. The HTTP embedding provider uses `urllib` from stdlib. Optional semantic search requires an external OpenAI-compatible embedding endpoint.

---

## Repository Layout

```
src/vertical_brain/
├── core/
│   ├── models.py             — All data classes (Chunk, Node, Link, StorageOperation, …); StrEnum types
│   ├── operations.py         — StorageOperationExecutor: validate + dispatch + apply + audit + Bronze dedup guard
│   ├── errors.py             — Typed domain errors (VerticalBrainError + subclasses)
│   ├── optimizer.py          — SimpleOptimizer: dedup + Silver compaction + decay + cross-namespace link discovery
│   ├── gold.py               — GoldAspect v2, parse/serialize, legacy GoldDocument
│   ├── search.py             — BrainSearch: lexical FTS + path ranking
│   ├── embedding_search.py   — EmbeddingSearch: cosine + persistent vector cache
│   ├── embedding_router.py   — EmbeddingRouter: route by embedding similarity + v1.7 Bronze/Silver deep-fallback (`find_candidates_with_fallback`)
│   ├── context_lock.py       — ContextLock: builds locked context capsules
│   ├── context_session.py    — ContextSession: map + search + lock orchestration
│   ├── namespace_map.py      — NamespaceMapBuilder: model-facing orientation map
│   ├── router.py             — LLMRouter + StorageModel: validate route decisions
│   ├── doctor.py             — Doctor: storage integrity checks
│   └── json_schema.py        — Stdlib JSON Schema validator (no external deps)
│
├── storage/
│   ├── protocol.py           — StorageProvider Protocol (runtime-checkable); includes has_active_chunk_with_hash
│   ├── derivations.py        — StorageDerivationsMixin: get_peer_paths / get_ancestors / tree_text
│   ├── sqlite_store.py       — SQLiteStore: WAL, FTS5, transactions, vector cache, dedup index, schema_version
│   ├── json_store.py         — JsonStore: human-readable files, dev/debug
│   └── thread_local_store.py — ThreadLocalSQLiteStoreProxy
│
├── llm/
│   ├── embedding.py          — EmbeddingProvider Protocol, Mock, HttpEmbeddingProvider
│   └── mock_llm.py           — MockLLM for testing
│
├── mcp/
│   ├── server.py             — MCP stdio server (JSON-RPC 2.0, NDJSON over stdio)
│   ├── tools.py              — MCP tool schemas (_TOOLS) + dispatch helpers
│   ├── read_path.py          — Read-path helpers: silver_confidence (v1.5) + semantic upgrade (v1.6), next_hint builders, apply_layer_bias, suggested_paths, is_semantic_provider
│   ├── ingest_handlers.py    — _IngestHandlers mixin: ingest session lifecycle
│   ├── ingest_protocol.py    — Ingest IRON RULES, splitting, answer_complete gates
│   └── extractors/           — Source text extraction: pdf, docx, doc, html, odt
│
└── cli/
    └── main.py               — `vb` CLI entry point
```

Test files mirror `src/` — there is one test file per module, plus cross-cutting tests (`test_occ.py`, `test_version_propagation.py`, etc.).

---

## Running Tests

```bash
pytest                                    # full suite (592 tests)
pytest tests/test_operations.py -v       # single file
pytest -k "occ or version" -v            # keyword filter
pytest --tb=short 2>&1 | tail -20        # summary view
```

All tests use `tmp_path` fixtures — no shared state, no cleanup required.

**Run with Python 3.11+.** The project requires 3.11 (`enum.StrEnum`). Running the suite under an older interpreter fails at collection — use a 3.11+ `python -m pytest`.

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

8. **Bronze dedup is enforced at the executor level, not the store level.** `_op_append_chunk` checks exact-hash duplicates via `store.has_active_chunk_with_hash()` (hard block — raises `DuplicateBronzeError`) and near-duplicates via `_find_similar_bronze()` (soft warning in `OperationResult.similar_bronze`). The check applies only to `layer="bronze"` — Silver and Gold are exempt. The composite index `idx_chunks_dedup (node_path, status, content_hash)` makes the hard block O(log n).

9. **`append_gold_aspect` requires an active Silver chunk at the same namespace.** The executor raises `GoldGroundingError` if no active Silver exists. Gold must be grounded in Silver — write Silver first via `update_silver`, then promote to Gold.

---

## Adding a New Operation

1. Add a member to the `OperationType` StrEnum in `models.py`.
2. Add a handler `_op_{name}(self, operation)` in `StorageOperationExecutor` in `operations.py`.
3. Register the handler in the `_APPLY_DISPATCH` dict (keyed by operation name).
4. If the operation needs operation-specific validation, extend `_validate_operation`.
5. Update the JSON Schema in `data/namespaces/model.json` (the `operation` enum).
6. Write a test in `tests/test_operations.py`.

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

Vectors are keyed by `(content_hash, model_name)`. The model name is recorded in the store schema (`set_embedding_schema`). If you change the embedding model, run the `vb reindex` CLI command to purge old vectors and re-embed (`EmbeddingSearch.trigger_reindexing(new_provider)` is the underlying call).

`EmbeddingSearch.__init__` raises `IncompatibleEmbeddingModelError` if the provider's model name doesn't match what's stored — this is intentional and should not be silenced.

---

## Read-Path Helpers (v1.5–v1.8)

`mcp/read_path.py` contains the small pure helpers wired into `server.py` to make the read path provider-agnostic. They have no I/O of their own — server.py supplies the inputs.

- **`silver_confidence(query, silver_item)`** → `"missing" | "low" | "weak" | "ok"`. Returns `weak` only when **lexical token overlap is below ~30%** — server.py then layers v1.6 semantic upgrade on top.
- **`semantic_silver_upgrade(lexical, query_vec, silver_vec, threshold=0.5)`** (v1.6) — **only** promotes `weak` → `ok`, never any other state. Any other state is returned unchanged. No-op when vectors are missing.
- **`is_semantic_provider(provider)`** — `True` for anything that isn't `MockEmbeddingProvider`. Used to gate `semantic_silver_upgrade` and to populate `semantic_endpoint` in `route` / `search_semantic` responses.
- **`route_next_hint(...)` / `read_context_next_hint(...)`** — build agent-facing hints; emitted only when the answer needs help (low Gold confidence, weak/missing Silver).
- **`apply_layer_bias(results)`** — stable reorder of semantic search hits to put Bronze evidence above Silver above Gold within the same score band. No result is dropped.
- **`suggested_paths(results)`** — unique namespaces in result order (post-bias), used as `read_context` suggestions in the `search_semantic` response.

**Where the v1.7 deep-fallback lives:** `EmbeddingRouter.find_candidates_with_fallback(text, fallback_threshold)` in `embedding_router.py`. It calls `find_candidates` first, then — only if the top Gold score is below `fallback_threshold` — runs `EmbeddingSearch` over Bronze/Silver and merges those namespaces in with `match_source="content_fallback"`. Negative `fallback_threshold` disables the rescue entirely.

**Read-path invariants (don't break):**

10. **Only `weak` is upgraded by cosine** — `semantic_silver_upgrade` must never promote `missing`/`low`/`ok`. Tested in `test_read_path_hints.py::test_semantic_upgrade_*`.
11. **`next_hint` is emitted only when help is needed** — `route_next_hint` returns `None` when Gold is confident; `read_context_next_hint` returns `None` when `silver_confidence == "ok"`. Don't add hints to the happy path.
12. **`apply_layer_bias` is stable and total** — same input order is preserved within a bucket, no result is filtered out. The bias only reorders.
13. **`route` response carries `match_source` for every candidate** — `"gold"` for Gold/path-token hits, `"content_fallback"` only for v1.7 rescue. Bumping anything else into `content_fallback` would mislead agents.
14. **`omitted_chunk_ids` only contains real chunk IDs** — Gold-aggregated items have no single `chunk_id`; they go into `omitted_items` only. The list must stay safe to feed into `list_chunks` or chunk-id lookups.
15. **`similar_bronze` returns `bm25_score`, not `score`** — FTS5/BM25 is unbounded and lower = more similar. The whole rest of the API uses `[0, 1]` cosine, so the v1.8 rename is load-bearing.

---

## SQLite Concurrency

- WAL mode: N concurrent readers + 1 writer
- One `SQLiteStore` per thread. Do not share instances across threads.
- For multi-threaded code, use `ThreadLocalSQLiteStoreProxy` — it creates one `SQLiteStore` per thread lazily.
- `transaction()` is reentrant: the outermost level drives `BEGIN`/`COMMIT`/`ROLLBACK`, nested levels use `SAVEPOINT` so an inner block can roll back independently.

---

## Commit Style

This repo uses conventional commit messages. Keep messages short and factual. Multi-line body when the change is non-trivial. No trailing "Co-authored" lines needed unless explicitly requested.

---

## Things to Avoid

- **Do not add external Python dependencies.** The zero-runtime-dependency constraint is intentional.
- **Do not bypass the executor.** Go through `apply` / `apply_batch`; never call `_op_*` handlers or `_apply_validated` directly from outside `StorageOperationExecutor`.
- **Raise typed errors, not bare `ValueError`.** Executor rejections use the `core/errors.py` hierarchy (`VerticalBrainError` and subclasses). It subclasses `ValueError`, so existing `except ValueError` keeps working.
- **Do not write raw strings to Gold.** Always use `append_gold_aspect` / `serialize_gold_aspects` — never mutate Gold chunks directly.
- **Do not share `SQLiteStore` across threads.** Use `ThreadLocalSQLiteStoreProxy`.
- **Do not read storage inside `_build_plan`.** The optimizer is pure over its snapshot input.
- **For agent-facing usage rules** (optimize, vacuum, rename_namespace, write layering) — see `AGENTS.md`.
