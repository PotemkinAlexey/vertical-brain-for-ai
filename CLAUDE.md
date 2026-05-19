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
│   ├── embedding.py          — EmbeddingProvider Protocol (v1.9: embed_batch, is_semantic, BaseEmbeddingProvider), Mock, HttpEmbeddingProvider (native batch endpoint)
│   ├── reranker.py           — RerankerProvider Protocol (v1.10 extension point; no default impl)
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
pytest                                    # full suite (654 tests)
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
- `TransactionalStorageProvider` for atomic batches with rollback
- `AuditableStorageProvider` to fan audit rows out to external systems (rows are still written to the operation_audit table — see invariant 22)

`EmbeddingSearch` and `EmbeddingRouter` detect these via `getattr` and degrade gracefully without them.

**Backend MUST round-trip `Chunk.metadata` and `Node.metadata` (v1.12).** Empty default is `{}`. JSON columns are the obvious mapping for SQL backends; document stores can persist as a nested map. Failure to round-trip will silently break `chunk_filter`-based ACL.

**Backend MAY ignore `metadata` semantics.** The open core neither indexes nor queries `metadata` — keys are opaque from the framework's perspective. Enterprise that wants `WHERE metadata->>'tenant_id' = ?` indexing is free to add it without coordinating with the open repository.

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

## Open-Core Extension Points (v1.9–v1.13)

These are the seams enterprise builds on without forking. Every one is
additive — the open core works with all defaults — and every one is
tested. Treat the signatures as frozen unless a major version bump.

### v1.9 — EmbeddingProvider Protocol

`vertical_brain/llm/embedding.py`. The `EmbeddingProvider` Protocol
declares `model_name`, `embed_dimension`, `is_semantic`, `embed(text)`,
and `embed_batch(texts)`. `BaseEmbeddingProvider` is the recommended
base class with safe defaults (`is_semantic = True`, `embed_batch =
N×embed`); only override what you need.

- **Cache key is `(content_hash, model_name)`.** Switching models without
  reindex returns mismatched vectors → handled by raising
  `IncompatibleEmbeddingModelError` at `EmbeddingSearch.__init__`.
- **`is_semantic = False` is the bag-of-words signal.** Read-path uses it
  via `is_semantic_provider` to gate semantic upgrades and to phrase the
  agent-facing "endpoint not configured" hint.
- **`embed_batch` is the collapse point for N×latency.** `EmbeddingSearch._resolve_vectors`
  issues one batch call per search for all cache-misses; providers without
  `embed_batch` fall back to N×embed automatically.

### v1.10 — RerankerProvider Protocol

`vertical_brain/llm/reranker.py`. Cross-encoder / re-scoring stage that
sits between cosine retrieval and layer bias. Pass to
`VerticalBrainMCP(reranker=...)` or directly to `EmbeddingSearch.search(reranker=...)` /
`ContextSession.search_locked_context_semantic(reranker=...)`.

- **Reranker MAY drop candidates, MUST NOT invent them.** Fabricated
  results (chunk_ids not in the input pool) are filtered out
  defensively in `_apply_reranker`.
- **Exceptions from the reranker are swallowed.** The read path must
  not break on a third-party service failure; cosine order is the
  fallback.
- **Layer bias runs AFTER the reranker.** Reranker refines scores;
  layer bias enforces the Bronze>Silver>Gold policy.
- **Pool size = `limit * 3`.** Wide enough for the cross-encoder to
  refine, narrow enough to stay under Cohere/Voyage per-call limits.

### v1.11 — chunk_filter callback

Optional `Callable[[Chunk], bool]` accepted by every read-path API:
`ContextLock.open_locked_context`, `BrainSearch.search`,
`EmbeddingSearch.search`, `EmbeddingRouter.find_candidates*`,
`ContextSession.search_locked_context*`. Wired into MCP via
`VerticalBrainMCP._current_chunk_filter()` (override returns the per-
request closure).

- **Default `None` = full visibility.** Zero behavioural change.
- **Filter applies BEFORE the reranker.** Hidden chunks never reach
  the cross-encoder.
- **`BrainSearch` resolves Chunk via `store.get_chunk(chunk_id)` post-
  search** because the FTS index can't evaluate Python callables;
  oversamples 3× so filtered-out results don't starve the returned set.
- **The filter is the natural counterpart to `Chunk.metadata` (v1.12)**
  — typical predicate is `lambda c: c.metadata.get("classification") in allowed`.

### v1.12 — Chunk.metadata / Node.metadata

Both dataclasses carry a `metadata: dict[str, Any] = field(default_factory=dict)`.
`ChunkInput` forwards a matching field. SQLite serializes via
`metadata_json TEXT NOT NULL DEFAULT '{}'`; legacy databases migrate
automatically on open.

- **The open core never reads `metadata`.** It only round-trips bytes —
  enterprise (and `chunk_filter`) own the semantics.
- **Defaults to `{}`, never `None`.** Read paths can safely call
  `.get(...)` without a None-guard.
- **Round-trip is tested for both backends.** `tests/test_metadata_slot.py`
  pins the contract including a hand-crafted legacy DB.

### v1.13 — before_tool / after_tool MCP hooks

`VerticalBrainMCP._before_tool(name, args, *, request_id)` and
`_after_tool(name, args, response, *, request_id)`. Default
pass-through; override in a subclass.

- **`_before_tool` runs AFTER tool-name existence check and BEFORE schema validation.**
  This is the order that lets the hook supply missing required args.
- **`_before_tool` MUST return a dict.** Non-dict return surfaces as -32603.
- **`ToolAccessDenied(message, *, code=-32004)` is the canonical denial.**
  Catch happens only in `_dispatch_tool`; the code defaults to the
  Vertical Brain-reserved -32004 and may be overridden outside the
  JSON-RPC reserved -32700..-32600 range.
- **`_after_tool` is SKIPPED on tool-body errors.** Error replies go
  straight back; non-critical after-hook failures should be swallowed
  internally by the hook author.
- **Args passed to `_before_tool` are a shallow copy.** Hook mutations
  do not leak back into JSON-RPC params on subsequent calls.

**Extension invariants (don't break):**

16. **`EmbeddingProvider.model_name` keys the vector cache.** Two
    providers with the same `model_name` MUST produce comparable
    vectors. Rename when you change the model.
17. **`is_semantic = False` ⇒ no semantic upgrade.** `is_semantic_provider`
    must continue to return False so `semantic_silver_upgrade` is gated off.
18. **Reranker output preserves SearchResult identity.** Only `score` is
    mutable; `chunk_id`/`path`/`layer` MUST come from the input pool.
19. **`chunk_filter=None` is identical to no kwarg.** Tested across all
    five read-path API entry points.
20. **`metadata` defaults to `{}`, not `None`.** All readers can assume
    a dict. Backend-level migrations enforce this.
21. **`_before_tool` runs before validation; `_after_tool` runs after
    success only.** Tested in `test_mcp_hooks.py`.
22. **The audit log row is written inside the same transaction as the
    operation.** `StorageOperationExecutor._log_audit` is invoked within
    the executor's transaction context; an `AuditableStorageProvider` that
    fans out to external systems must respect this — fire-and-forget
    external writes are NOT a substitute for the SQL row.

---

## Enterprise Tenant-Prefix Pattern

The open core has no native multi-tenancy. The supported pattern when
you build a multi-tenant deployment on top is:

1. **Reserve the top-level namespace per tenant.** Path roots like
   `tenant_acme/WORK/...`, `tenant_globex/PERSONAL/...`. The path is
   the tenant boundary — `root_path` filtering then becomes the
   isolation primitive.
2. **Inject the tenant root in `_before_tool`.** Override the hook to
   set `args["root_path"] = f"tenant_{claim.tenant_id}"` (or wrap an
   existing `root_path`) for every read tool. For writes, validate the
   `target_path` starts with the tenant prefix and raise
   `ToolAccessDenied` otherwise.
3. **Stash the tenant id on the request via contextvars**, then
   consume it in `_current_chunk_filter` to layer a defence-in-depth
   `chunk.metadata["tenant_id"] == claim.tenant_id` check.
4. **Set `Chunk.metadata["tenant_id"]` at write time** through a
   `_before_tool` override that adds it to `args["chunk"]["metadata"]`
   (or via a custom executor that injects metadata in
   `_chunk_from_input`).
5. **Audit log enrichment** — extend the audit row via an
   `AuditableStorageProvider` to include `tenant_id` and the JWT subject.

The open core enforces none of this; it provides the seams. The
combination of `root_path` (path-level scope), `_before_tool` (request
rewrite), `chunk_filter` (row-level ACL), and `metadata` (per-row
classification) is what enterprise composes into a tenant boundary.

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
