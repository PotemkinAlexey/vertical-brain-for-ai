# MCP Tools Reference

The Vertical Brain MCP server speaks JSON-RPC 2.0 over stdio with Content-Length framing (LSP-style). It exposes the following tools to Claude and other MCP clients.

All parameters are optional unless marked **required**.

---

## Orientation

### `session_start`

Call this at the beginning of every session. Returns a compact text prompt containing all namespaces, Gold summaries, chunk counts, and link handles — enough to orient without flooding the context.

| Parameter | Type | Description |
|-----------|------|-------------|
| `root_path` | string | Limit map to this namespace subtree |
| `max_depth` | integer | Maximum depth relative to `root_path` |
| `summary_chars` | integer | Max Gold characters per node (default 200) |

---

### `namespace_map`

Returns the full namespace map as JSON. Useful when you need programmatic access to the structure.

| Parameter | Type | Description |
|-----------|------|-------------|
| `root_path` | string | Limit map to this namespace subtree |
| `max_depth` | integer | Maximum depth relative to `root_path` |
| `summary_chars` | integer | Max Gold characters per node |

**Response fields per node:** `path`, `depth`, `chunk_count`, `active_chunk_count`, `subtree_chunk_count`, `link_count`, `gold_summary`, `children`, `link_handles`.

---

### `list_chunks`

List all chunks at a namespace path. Returns full content. Use this to read what is actually stored before making decisions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | **required** Namespace path |
| `include_stale` | boolean | Include stale and superseded chunks (default false) |
| `layer` | `bronze`\|`silver`\|`gold` | Filter by layer (omit for all) |

---

### `read_context`

Open a locked context capsule for a namespace path. Returns chunk content (Gold → Silver → Bronze priority), ancestor Gold summaries, and link handles. Budget-capped to prevent context overflow.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | **required** Namespace to open |
| `include_ancestors` | boolean | Include ancestor Gold summaries (default true) |
| `link_expansion` | `handles_only`\|`expanded`\|`none` | How to expose horizontal links (default `handles_only`) |
| `max_items` | integer | Maximum items in the context capsule |

---

## Search

### `search`

Lexical full-text search across active chunks. Uses SQLite FTS5 (or prefix matching if FTS5 is unavailable). Returns ranked results with path, score, and content snippet.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | **required** Search terms |
| `root_path` | string | Limit search to this namespace subtree |
| `limit` | integer | Maximum results (default 10) |
| `include_stale` | boolean | Include stale/superseded chunks (default false) |

---

### `search_semantic`

Semantic similarity search using embedding cosine similarity. Requires the MCP server to be started with `--embedding-url`. Vectors are cached persistently.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | **required** Query text |
| `root_path` | string | Limit search to this namespace subtree |
| `limit` | integer | Maximum results (default 10) |
| `threshold` | number | Minimum cosine similarity score [0, 1] (default 0.0) |

---

### `context_search`

Lexical search followed by opening locked context capsules for the top-ranked namespaces. The most efficient way to find and read relevant content in one call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | **required** Search terms |
| `root_path` | string | Limit search to this namespace subtree |
| `search_limit` | integer | Maximum candidate handles from search |
| `context_limit` | integer | Maximum namespaces to open context for |
| `items_per_context` | integer | Maximum items per locked context |
| `include_ancestors` | boolean | Include ancestor Gold summaries (default true) |

---

### `context_search_semantic`

Same as `context_search` but uses embedding similarity instead of lexical search.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | **required** Query text |
| `root_path` | string | Limit search to this namespace subtree |
| `search_limit` | integer | Maximum candidate handles |
| `context_limit` | integer | Maximum namespaces to open |
| `items_per_context` | integer | Maximum items per context |
| `include_ancestors` | boolean | Include ancestor Gold summaries (default true) |
| `threshold` | number | Minimum cosine similarity score |

---

### `route`

Find the best-matching namespaces for a piece of text by comparing its embedding against Gold chunks. Returns ranked candidates with path and Gold summary.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | string | **required** Text to route |
| `limit` | integer | Maximum candidates (default 5) |
| `threshold` | number | Minimum similarity score (default 0.0) |

---

## Write

### `append_chunk`

Write a new chunk to a namespace. Creates the node chain if it does not exist.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | **required** Target namespace |
| `content` | string | **required** Chunk text |
| `layer` | `bronze`\|`silver`\|`gold` | Layer (default `bronze`) |
| `content_type` | `fact`\|`decision`\|`question`\|`note`\|`code`\|`artifact`\|`correction` | |
| `source` | string | Who produced this (e.g. `model`, `user`) |
| `confidence` | number | Quality signal [0, 1] (default 1.0) |

---

### `append_gold_aspect`

Add or refresh a semantic label in the Gold summary of a namespace. Exact-text duplicates refresh `updated_at` without creating a new aspect. Returns `overflow_path` if a new sibling namespace was created (when the 20-aspect limit is reached).

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | **required** Target namespace |
| `aspect` | string | **required** The semantic label text |

> **Naming note:** This MCP tool accepts the argument as `aspect`. The underlying `StorageOperation` field is `gold_aspect`. When building operation batches for the `operations` tool or `vb operation apply`, use `gold_aspect`.

---

### `create_link`

Create a horizontal link between two namespaces. Links appear as handles in locked context capsules and can be expanded on demand.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source_path` | string | **required** Source namespace |
| `target_path` | string | **required** Target namespace |
| `link_type` | string | **required** e.g. `peer`, `reference`, `derived_from`, `depends_on` |
| `reason` | string | **required** Human-readable explanation |

---

### `mark_stale`

Retire chunks at a namespace. Pass specific `chunk_ids` to target individual chunks, or omit to mark all active non-Gold chunks at the path.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | **required** Namespace |
| `chunk_ids` | string[] | Specific chunk IDs (omit for all active non-Gold) |
| `reason` | string | Why these chunks are stale (goes to audit log) |

---

### `batch_append`

Write multiple chunks in one call. All-or-nothing when using the SQLite backend.

| Parameter | Type | Description |
|-----------|------|-------------|
| `chunks` | object[] | **required** Array of chunk objects (each needs `path` and `content`) |

Each chunk object accepts: `path` (required), `content` (required), `layer`, `content_type`, `source`, `confidence`.

---

### `session_end`

Write a session summary as a Silver/note chunk. Call at the end of each session to persist what was learned.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | **required** Namespace to write the summary to |
| `summary` | string | **required** What was learned or done this session |
| `gold_aspect` | string | Optional semantic label to append to Gold |

---

### `operations`

Apply a full `StorageOperationBatch` JSON. The batch is validated against the storage model schema before any I/O. If validation fails, the entire batch is rejected with a structured error.

| Parameter | Type | Description |
|-----------|------|-------------|
| `payload` | object | **required** A `StorageOperationBatch` JSON object |

See [04_operations_reference.md](04_operations_reference.md) for the full batch schema.

---

## Maintenance

### `optimize`

Run the compaction optimizer on a namespace branch. Detects exact duplicates, compacts multiple Bronze/Silver chunks into canonical Silver summaries, and decays old unlinked content.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | **required** Namespace prefix to optimize |

---

### `doctor`

Run storage integrity checks and return a list of issues (orphan links, invalid fields, staging items older than 3 days, etc.).

No parameters.

---

## Protocol Notes

- The server speaks JSON-RPC 2.0 with Content-Length headers (same framing as LSP)
- All tool calls return `content: [{type: "text", text: "..."}]` on success
- Errors return `isError: true` with the error message in `content[0].text`
- The server is single-threaded; concurrent tool calls from one session are serialized
