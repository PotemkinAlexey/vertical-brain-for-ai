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

Find the best-matching namespaces for a piece of text by comparing its embedding against individual Gold aspects. Returns ranked candidates with path and the best matching Gold aspect. Aspect vectors are persisted in `vector_cache` and reused across router instances.

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

**Bronze dedup (layer `bronze` only):**

- **Hard block:** if an active Bronze chunk with byte-for-byte identical content already exists at `path`, the call returns an error. Call `mark_stale` on the old chunk first if the fact has changed, then retry.
- **Soft warning:** if the write succeeds but similar active Bronze chunks exist at `path`, the response includes `similar_bronze` — a list of up to 3 matching chunks with snippets. Review them and `mark_stale` any that the new chunk supersedes.

Silver and Gold are exempt from both checks.

---

### `append_gold_aspect`

Add or refresh a short semantic routing tag in the Gold index of a namespace. Exact-text duplicates refresh `updated_at` without creating a new aspect. Returns `overflow_path` if a new sibling namespace was created (when the 20-aspect limit is reached). Returns `aspect_too_long: true` if the aspect exceeds 150 characters.

> **Prerequisite:** An active Silver chunk must exist at `path`. Call `update_silver` first if it doesn't.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | **required** Target namespace |
| `aspect` | string | **required** Short search tag; target ~30–100 characters |

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

Persist a session summary following Bronze → Silver → Gold layering. Always writes a Bronze note chunk. Also writes Silver when `notes` is provided or when `gold_aspect` is requested. If only `summary` is passed without `notes` or `gold_aspect`, Silver promotion remains the agent's responsibility.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | **required** Namespace to write the summary to |
| `summary` | string | **required** Refined Silver summary of what was learned or done |
| `notes` | string | Optional raw Bronze notes — detailed facts and observations |
| `gold_aspect` | string | Optional durable insight to append to Gold |

---

### `update_silver`

Atomically replace the active Silver summary for a namespace. Supersedes the old Silver chunk and writes a new one. Enforces OCC — the agent must pass the ID of the Silver chunk it read before synthesizing. Backed by the `update_silver` `StorageOperation`.

> **First Silver:** if no active Silver exists yet, create it with `append_chunk(layer="silver")`. After that, all Silver updates must use `update_silver`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | **required** Namespace to update |
| `new_content` | string | **required** Complete rewritten Silver summary |
| `current_silver_id` | string | **required** ID of the active Silver chunk you read before synthesizing |
| `source_chunk_ids` | array | Optional IDs of Bronze chunks incorporated into this Silver (used for lineage) |
| `confidence` | number | Quality signal [0, 1] (default 1.0) |
| `reasoning_summary` | string | Why this Silver was rewritten (goes to audit log) |

Returns `{"status": "applied", "chunk_id": "<new silver id>"}`.

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

### `vacuum`

Databricks-style maintenance cleanup. Dry-run by default. Physically purges old inactive chunks and service debris when `dry_run=false`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `retention_hours` | number | Retention window before an inactive chunk is eligible (default 168) |
| `dry_run` | boolean | Preview candidates without deleting (default true) |
| `force` | boolean | Required to apply retention below 168 hours |
| `include_immutable` | boolean | Also purge immutable inactive chunks; requires `force=true` when applying |
| `prune_empty_nodes` | boolean | Remove namespace nodes with no chunks, links, or children (default true) |
| `prune_vector_cache` | boolean | Remove orphan embedding vectors while preserving active chunk and Gold aspect vectors (default true) |
| `reclaim_space` | boolean | Run SQLite `VACUUM` after purging |

The tool returns counts, candidate metadata, and checkpoint status. Use CLI
`vb vacuum --apply --backup FILE` when you need an automatic backup before
deletion.

---

### `ingest_file`

Registers a document for guided ingestion. Creates an ingest session, splits the content into service chunks, and returns a `session_key` with a numbered chunk list and the full ingestion protocol inline. For PDFs, use `source_path` — the server runs `pdftotext`. For all other formats, the agent reads the file and passes `content`. The agent must follow the inline protocol to completion before responding to the user.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source_path` | string | Local path to the original file; required for PDFs |
| `content` | string | Full text content of the attached non-PDF file |
| `file_name` | string | Original file name, e.g. `MT103.txt`; defaults to `source_path` basename |
| `expected_sha256` | string | Optional SHA-256 expected for the text payload; mismatches are rejected |
| `expected_size_bytes` | integer | Optional UTF-8 byte count expected for the text payload; mismatches are rejected |
| `authority` | string | Issuing authority (e.g. `SWIFT`, `ISO`). Inferred from content if omitted |
| `doc_slug` | string | Short namespace identifier. Defaults to filename without extension |

---

### `get_service_chunk`

Reads the full content of one service chunk from an active ingest session. Call this for each chunk ID returned by `ingest_file` before writing Bronze facts.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_key` | string (required) | Session key returned by `ingest_file` |
| `chunk_id` | string (required) | Chunk ID from the `ingest_file` chunk list |

---

### `mark_service_chunk`

Marks a service chunk as processed. Call with `status='extracted'` after writing Bronze facts, or `status='skipped'` with a reason for boilerplate or irrelevant content. All chunks must be marked before `finish_bronze_extraction` will succeed.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_key` | string (required) | Session key |
| `chunk_id` | string (required) | Chunk ID |
| `status` | string (required) | `"extracted"` or `"skipped"` |
| `skip_reason` | string | Required when `status='skipped'` |

---

### `finish_bronze_extraction`

Validates that all service chunks have been marked extracted or skipped. Returns an error listing pending chunk IDs if any remain. Transitions the session to `BRONZE_COMPLETE`. Call this before writing Silver.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_key` | string (required) | Session key |

---

### `complete_ingest`

Completes the ingest session after Silver has been written. Validates that `finish_bronze_extraction` was called first. Cleans up service chunks from memory.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_key` | string (required) | Session key |

---

### `ingest_url`

Fetches a URL and registers its content as a source document. Supports plain text, Markdown, JSON, YAML, and HTML (tags are stripped with stdlib `html.parser`). No external dependencies.

| Parameter | Type | Description |
|-----------|------|-------------|
| `url` | string (required) | `http` or `https` URL to fetch |
| `authority` | string | Issuing authority. Defaults to the URL hostname |
| `doc_slug` | string | Short namespace identifier. Defaults to the last URL path segment |

The returned text contains the URL, SHA-256 of the fetched content, size, suggested `SOURCES` namespace, and the extracted text. The agent then applies the file ingestion steps: register source → extract Bronze → write Silver.

---

## Protocol Notes

- The server speaks JSON-RPC 2.0 over stdio with newline-delimited JSON (one object per line)
- All tool calls return `content: [{type: "text", text: "..."}]` on success
- Errors return `isError: true` with the error message in `content[0].text`
- The server is single-threaded; concurrent tool calls from one session are serialized
