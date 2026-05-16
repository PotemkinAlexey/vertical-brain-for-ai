# Vertical Brain — Architecture

Vertical Brain is a personal knowledge store designed around one constraint: **a model should never see more than it needs to answer a specific question.** Everything else follows from that.

---

## Design Principles

1. **Namespace isolation first.** Knowledge lives in vertical paths. A context capsule opened at `WORK/DataArt/Databricks` includes only that subtree and its ancestors — never peer branches or unrelated domains.

2. **Map before locking.** The model always starts with a compact orientation map, then locks a specific namespace. This prevents context stuffing and makes sessions reproducible.

3. **Operations, not raw writes.** Every mutation is a `StorageOperation` validated against JSON Schema. Validation happens before any I/O. Invalid operations are rejected, not silently truncated.

4. **Layered quality.** Raw input (Bronze) flows through compaction to canonical facts (Silver) to stable summaries (Gold). Only Gold is shown in orientation maps; Silver and Bronze are available in locked context.

5. **Deterministic optimizer.** The optimizer performs a bounded snapshot read phase: branch chunks once, branch node version once, and links once only when decay is enabled. Planning is pure over that snapshot — no further I/O. The resulting batch is applied transactionally. No surprises.

---

## Data Model

### Node

A node is a namespace path. It is created automatically when a chunk is written to it.

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Stable identifier |
| `path` | string | Slash-separated path, e.g. `WORK/DataArt` |
| `name` | string | Last segment of the path |
| `parent_path` | string\|null | Parent path (null for roots) |
| `node_type` | string | `namespace` or `root` (default `"default"`) |
| `version` | int | Monotonically increasing; incremented on every mutation |
| `is_dirty` | bool | True if content changed since last Gold rebuild |
| `created_at` | ISO 8601 | UTC creation time |
| `updated_at` | ISO 8601 | UTC last mutation time |

> **Gold summaries are stored as Gold-layer chunks, not as a field on Node.** The `Node` dataclass has no `gold_summary` field. When the namespace map or session prompt shows a Gold summary, it is read from the active Gold-layer chunk at that path. The SQLite backend maintains an internal `gold_summary` column for fast orientation reads, but this is an implementation detail of the store and not part of the `Node` protocol.

### Chunk

A chunk is a single piece of content at a node. Multiple chunks can coexist at the same node in different layers and states.

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Stable identifier |
| `node_path` | string | The namespace this chunk belongs to |
| `layer` | `bronze`\|`silver`\|`gold` | Knowledge quality layer |
| `content` | string | The actual text |
| `content_type` | enum | `fact`, `decision`, `question`, `note`, `code`, `artifact`, `correction` |
| `status` | enum | `active`, `stale`, `superseded`, `legacy`, `contradicted` |
| `source` | string | Who produced this: `user`, `model`, `optimizer:namespace_compaction`, etc. |
| `confidence` | float [0,1] | Routing/quality signal |
| `lineage` | list[UUID] | IDs of source chunks this was distilled from |
| `content_hash` | string | SHA-256 of content, used for deduplication and vector cache keys |
| `decay_factor` | float | Per-chunk decay rate (overrides global if < 1.0) |
| `valid_from` | ISO 8601 | When this chunk became active |
| `valid_to` | ISO 8601\|null | Auto-set when status transitions to stale/superseded/legacy/contradicted |
| `created_at` | ISO 8601 | |
| `updated_at` | ISO 8601 | |

### Link

A link is a horizontal connection between two namespace paths.

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Stable identifier |
| `source_path` | string | Source namespace |
| `target_path` | string | Target namespace |
| `link_type` | string | `related`, `depends_on`, `supersedes`, `peer`, etc. |
| `reason` | string | Human-readable description |

### GoldAspect (v2 format)

Gold aspects are stored as the content of an active Gold-layer chunk at the node's path. The chunk content is a v2 JSON object:

```json
{
  "aspects": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "text": "Delta Lake Z-ordering reduces scan time by clustering rows.",
      "updated_at": "2025-01-15T10:30:00Z"
    }
  ]
}
```

- `id` is stable across rewrites — updating the same aspect refreshes `updated_at` but keeps the UUID
- Max 20 aspects per node; overflow creates a sibling namespace `{path}_2`, `{path}_3`, etc.

There are two distinct Gold mechanisms in the codebase:

- **`append_gold_aspect` / `GoldAspect`** — the lightweight incremental path. Adds or refreshes a single semantic label. Used by the model and MCP tools to annotate a namespace during normal operation.
- **`GoldDocument` / `GoldBuilder` / `LlmGoldBuilder`** — the structured distillation path. Builds a full Gold document from a set of Silver chunks, with typed `facts`, `entities`, and `rules`. Each `GoldFact` carries `source_silver_ids` that must map to real Silver chunks (hallucinated IDs are rejected). Use this path for batch Gold rebuilds driven by an LLM.

---

## Knowledge Layers

```
Bronze  ──→  Silver  ──→  Gold
(raw)       (canonical)   (stable)
```

### Bronze

What comes in. Raw user notes, model observations, questions, code snippets. Multiple bronze chunks can exist at the same node — they accumulate until the optimizer runs.

### Silver

Canonical facts. The optimizer compacts multiple Bronze chunks at a node into a single Silver summary, preserving lineage. Silver chunks de-duplicate by content hash.

### Compaction (Bronze → Silver)

`SimpleOptimizer.optimize_branch(path)` performs a bounded snapshot read (branch chunks, branch node version, and links when decay is enabled), then runs three passes over that snapshot with no further I/O:

1. **Exact dedup** — mark stale any active chunk whose `(node_path, content_hash)` already exists
2. **Namespace compaction** — for each node with 2+ active non-Gold chunks lacking lineage, create one Silver summary and supersede the originals
3. **Confidence decay** — mark stale any old Bronze/Silver chunk whose decayed confidence falls below `stale_threshold`

The entire batch is applied in a single transactional tick. If optimistic concurrency detects a concurrent write (`branch_path` + `start_version`), it raises `OptimisticLockException`.

### Gold

Stable semantic labels. Gold aspects are managed explicitly via `append_gold_aspect` — they are not overwritten wholesale. Each aspect has a stable UUID so downstream indexes can track identity across rewrites.

---

## Read Path

```
Model
  │
  ▼
session_start / namespace_map
  │   Returns compact orientation: paths, Gold summaries,
  │   chunk counts, link handles. No raw content.
  │
  ▼
search / context_search
  │   Lexical (FTS5) or semantic (cosine similarity) search.
  │   Returns ranked handles — path + score + snippet.
  │
  ▼
read_context / context_search
  │   Opens a locked context capsule for the target path:
  │   - Full content at the node (Gold → Silver → Bronze priority)
  │   - Ancestor Gold summaries
  │   - Link handles (not expanded)
  │   Budget enforced: items_per_context cap.
  │
  ▼
context_expand (optional)
      Follows a link handle to open its target's locked context.
      Same budget rules apply.
```

**Why handles, not content?** Link handles let the model decide whether to expand. Automatic expansion would flood the context with potentially irrelevant cross-domain content.

---

## Write Path

```
Model emits StorageOperation or StorageOperationBatch JSON
  │
  ▼
JSON Schema validation (json_schema.py)
  │   Rejects malformed operations before any I/O.
  │
  ▼
StorageOperationExecutor.apply_batch()
  │   - Opens transaction (SQLite: WAL savepoint)
  │   - OCC check inside transaction: branch_path + start_version
  │   - Applies each operation in order
  │   - Marks ancestors dirty + increments version
  │   - Commits atomically
  │
  ▼
Storage (SQLiteStore or JsonStore)
  │
  ▼
Audit log (operation_audit table / operation_audit.jsonl)
```

### Optimistic Concurrency Control

`StorageOperationBatch` carries optional `branch_path` and `start_version`. If set, the executor checks `node.version == start_version` inside the transaction before any mutations. A stale version raises `OptimisticLockException`.

The optimizer always stamps `branch_path`/`start_version` from the snapshot it read, so concurrent writes are detected and rejected cleanly.

---

## Component Map

```
vertical_brain/
├── core/
│   ├── models.py             — Data classes: Chunk, Node, Link, StorageOperation, …
│   ├── operations.py         — StorageOperationExecutor: validate + apply
│   ├── optimizer.py          — SimpleOptimizer: dedup + compaction + decay
│   ├── gold.py               — GoldAspect, parse/serialize, LlmGoldBuilder
│   ├── search.py             — BrainSearch: lexical FTS + path ranking
│   ├── embedding_search.py   — EmbeddingSearch: cosine similarity + vector cache
│   ├── embedding_router.py   — EmbeddingRouter: namespace routing by embedding
│   ├── context_lock.py       — ContextLock: builds locked context capsules
│   ├── context_session.py    — ContextSession: map + search + lock orchestration
│   ├── namespace_map.py      — NamespaceMapBuilder: model-facing orientation map
│   ├── router.py             — LLMRouter: validate route decisions from models
│   ├── doctor.py             — Doctor: storage integrity checks
│   └── json_schema.py        — Stdlib JSON Schema validator (no external deps)
│
├── storage/
│   ├── protocol.py           — StorageProvider Protocol (runtime-checkable)
│   ├── sqlite_store.py       — SQLiteStore: WAL, FTS5, transactions, vector cache
│   ├── json_store.py         — JsonStore: human-readable files, dev/debug
│   └── thread_local_store.py — ThreadLocalSQLiteStoreProxy: per-thread instances
│
├── llm/
│   ├── embedding.py          — EmbeddingProvider Protocol, Mock, HttpEmbeddingProvider
│   └── mock_llm.py           — MockLLM: canned responses for testing
│
├── mcp/
│   └── server.py             — MCP stdio server (JSON-RPC 2.0, LSP framing)
│
└── cli/
    └── main.py               — `vb` CLI entry point
```

---

## Storage Protocol

`StorageProvider` is a `runtime_checkable` Protocol in `storage/protocol.py`. Both `SQLiteStore` and `JsonStore` implement it. You can add a new backend by implementing the protocol — no base class needed.

Core methods:

```python
# Nodes
ensure_node(path: str) -> Node          # create-or-get a node by path
get_node(path: str) -> Node | None
update_node(node: Node) -> Node
list_nodes() -> list[Node]
get_ancestors(path: str) -> list[str]

# Chunks
save_chunk(chunk: Chunk) -> Chunk
update_chunk(chunk: Chunk) -> Chunk
get_chunks_by_path(path: str, include_children: bool = False) -> list[Chunk]
list_chunks() -> list[Chunk]

# Links
save_link(link: Link) -> Link
get_link(link_id: str) -> Link | None
get_peer_links(path: str) -> list[Link]
get_peer_paths(path: str) -> list[str]
list_links() -> list[Link]

# Utilities
tree_text() -> str
```

Optional extension protocols:
- `VectorCacheStorageProvider` — `get_vector`, `set_vector`, `delete_vectors_for_model`
- `EmbeddingSchemaStorageProvider` — `get_embedding_schema`, `set_embedding_schema`

`EmbeddingSearch` and `EmbeddingRouter` detect these extensions at runtime via `getattr` and degrade gracefully when absent.

---

## Concurrency Model

### SQLite

- WAL mode: N concurrent readers + 1 writer, no reader/writer blocking
- One `SQLiteStore` per thread — creating multiple instances on the same file is safe for reads, but concurrent writes from multiple instances will contend on the WAL lock
- Use `ThreadLocalSQLiteStoreProxy` in multi-threaded applications; it lazily creates one `SQLiteStore` per thread

### JSON

- Not thread-safe. For development only.

---

## Embedding Vector Cache

Vectors are stored keyed by `(content_hash, model_name)`:
- SQLite: `vector_cache` table (content_hash, model_name, vector_json, created_at)
- JSON: `vector_cache.json` flat dict

On `EmbeddingSearch` init, the stored model name is compared against the provider's `model_name`. Mismatches raise `IncompatibleEmbeddingModelError`. Switching providers via `trigger_reindexing(new_provider)` purges old vectors, updates the schema, and re-embeds all active chunks.

---

## Key Invariants

These invariants are preserved by the executor and tested explicitly:

1. **A chunk's `valid_to` is set** when its status transitions to `stale`, `superseded`, `legacy`, or `contradicted`.
2. **`node.version` increments** on every mutation at that node or any descendant.
3. **`node.is_dirty` is set** whenever a chunk is added or modified at that node.
4. **Compacted Silver chunks carry `lineage`** listing all source chunk IDs.
5. **Gold aspects are upserted by text** — exact-text duplicates refresh `updated_at`, never duplicate.
6. **OCC check is inside the transaction** — a version mismatch rolls back the entire batch.
7. **`rename_namespace` is atomic** — wrapped in `with self.transaction()` in SQLite.

---

## Known Limitations

- **No cross-store queries.** Each store is independent; federation is not implemented.
- **No real LLM routing.** `LLMRouter` expects JSON responses; plug in a real LLM via the response file mechanism or by subclassing.
- **Gold compaction is manual.** `append_gold_aspect` is called explicitly; there is no background Gold promoter.
- **FTS5 requires SQLite with FTS5 compiled in.** Falls back to prefix search if unavailable.
- **Python 3.11+.** Uses `Self` and structural pattern matching in places.
