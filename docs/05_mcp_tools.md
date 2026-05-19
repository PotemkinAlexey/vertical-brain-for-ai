# MCP Tools Reference

The Vertical Brain MCP server speaks JSON-RPC 2.0 over stdio with newline-delimited JSON (one JSON object per line). It exposes the following tools to Claude and other MCP clients.

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
| `content_type` | `fact`\|`reference`\|`decision`\|`question`\|`note`\|`code`\|`artifact`\|`correction` | |
| `source` | string | Who produced this (e.g. `model`, `user`) |
| `confidence` | number | Quality signal [0, 1] (default 1.0) |
| `immutable` | boolean | Protect Bronze reference material from dedup/size guards and ordinary stale/vacuum operations |

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

Write multiple chunks in one call, up to 10 chunks per call. All-or-nothing when using the SQLite backend.

| Parameter | Type | Description |
|-----------|------|-------------|
| `chunks` | object[] | **required** Array of chunk objects (each needs `path` and `content`) |

Each chunk object accepts: `path` (required), `content` (required), `layer`, `content_type`, `source`, `confidence`, `immutable`.

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

Run the compaction optimizer. With `path`: snapshot one branch (`optimize_branch`), then `discover_links` under that root. **Without `path`**: full-lake sweep (`optimize_all`) — walks each namespace via `get_chunks_by_path` (not `list_chunks()`), then cross-namespace link discovery using one representative Silver per namespace.

**Link discovery scale:** For P namespaces, embeddings are compared pairwise only when P ≤ 256 (brute force). Above that, random-hyperplane LSH (`vector_lsh.py`, stdlib-only) proposes candidate pairs, then exact cosine verification against the similarity threshold. Report lines include `[lsh: …]` or `[brute_force: …]` stats (namespace count, candidates, matches).

Detects exact duplicates, compacts Bronze/Silver variants into canonical Silver, and decays old unlinked content when decay is configured.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | Namespace prefix to optimize; omit for global `optimize_all` |

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

Starts a **stateful ingest session** with inline **IRON RULES** (server-enforced). Default `mode=answer_complete`: the document must be answerable from brain alone after ingest — the source file will not exist later.

For binary formats (`.pdf`, `.docx`, `.doc`, `.html`, `.htm`, `.odt`), use `source_path` — the server extracts text internally. For plain-text files, the agent reads the file and passes `content`. Passing binary-format content directly is rejected to prevent summarized payloads. Do not respond to the user until `complete_ingest` succeeds.

**answer_complete gates** (thresholds vary by `depth`): `[INVENTORY]` Bronze, Bronze facts, Silver (either `chunk_id` citations or sub-namespace path references), skip/extracted ratio limits, `submit_inventory_probes`, and `complete_ingest` storage verification. The session header shows the exact thresholds for the chosen depth.

**Depth presets:**

| `depth` | max skip | min extracted | min coverage | min probes |
|---------|----------|---------------|--------------|------------|
| `quick` | 50% | 30% | 60% | 0 (optional) |
| `standard` (default) | 25% | 50% | 90% | 3 |
| `thorough` | 10% | 75% | 95% | 5 |

**Recommended ingest flow for structured documents:**
1. Scan `section_header` service chunks → plan sub-namespace map (`SOURCES/{slug}/section-slug/`)
2. Write Bronze verbatim per section into sub-namespaces using `get_service_chunks` + `batch_mark_service_chunks`
3. Silver per sub-namespace = `[chunk_id] one-line summary` index; root Silver = `[SOURCES/{slug}/section] description`

Service chunk types: `splittable` (prose), `atomic` (code/table block, write verbatim), `section_header` (document heading — mark extracted, use to identify section boundaries).

`force=true` marks all prior chunks in the target namespace stale (including immutable) before starting a new session. Ingest sessions persist in SQLite until `complete_ingest`.

Use `mode=routing` only for discoverability-only ingests (lighter rules).

Source documents are stored under `SOURCES/{slug}`. `authority` is metadata in the artifact, not in the path.

| Parameter | Type | Description |
|-----------|------|-------------|
| `mode` | string | `answer_complete` (default), `routing`, or `audit` |
| `depth` | string | `quick`, `standard` (default), or `thorough` — sets skip/coverage thresholds |
| `source_path` | string | Local path to the original file; required for binary formats (`.pdf`, `.docx`, `.doc`, `.html`, `.htm`, `.odt`) |
| `content` | string | Full text content of the attached plain-text file |
| `file_name` | string | Original file name, e.g. `MT103.txt`; defaults to `source_path` basename |
| `expected_sha256` | string | Optional SHA-256 expected for the text payload; mismatches are rejected |
| `expected_size_bytes` | integer | Optional UTF-8 byte count expected for the text payload; mismatches are rejected |
| `authority` | string | Issuing authority (e.g. `SWIFT`, `ISO`), stored as metadata |
| `doc_slug` | string | Short namespace identifier. Defaults to filename without extension |
| `force` | boolean | Wipe the target namespace (including immutable chunks) and cancel prior sessions before starting (default false) |

---

### `submit_inventory_probes`

Before `complete_ingest` in `answer_complete` mode, submit probes that map inventory lines to active Bronze `chunk_id`s. Minimum count and required coverage depend on `depth` — the session header shows the exact thresholds. For `quick` depth, probes are optional. For `standard`, minimum is `max(3, ceil(30% × inventory lines))` with 90% coverage. For `thorough`, minimum is `max(5, ceil(50% × inventory lines))` with 95% coverage. Probes accumulate across calls — re-submitting the same item updates its `chunk_ids`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_key` | string (required) | Active ingest session |
| `probes` | array (required) | `{item, chunk_ids[]}` per probe |

---

### `get_service_chunks`

Reads up to 10 service chunks in one call (by `chunk_ids` or `start_index` + `count`). Prefer batching to reduce round-trips.

### `get_service_chunk`

Reads one service chunk from an active ingest session.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_key` | string (required) | Session key returned by `ingest_file` |
| `chunk_id` | string (required) | Chunk ID from the `ingest_file` chunk list |

---

### `mark_service_chunk`

Marks a service chunk as processed. Call with `status='extracted'` after writing Bronze facts, or `status='skipped'` only for empty/formatting noise, boilerplate, irrelevant text, or duplicates already represented in Bronze. All chunks must be marked before `finish_bronze_extraction` will succeed.

`skipped` is not a shortcut for closing an ingest session. The server rejects skip reasons that describe bulk-closing or completing ingest instead of a content-specific reason.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_key` | string (required) | Session key |
| `chunk_id` | string (required) | Chunk ID |
| `status` | string (required) | `"extracted"` or `"skipped"` |
| `skip_reason` | string | Required when `status='skipped'` |

---

### `batch_mark_service_chunks`

Marks multiple service chunks as extracted or skipped in a single call. Use after writing Bronze for an entire document section to avoid per-chunk round-trips. Max 50 marks per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_key` | string (required) | Session key |
| `marks` | array (required) | Array of `{chunk_id, status, skip_reason?}` objects (max 50) |

Each mark object follows the same rules as `mark_service_chunk`: `status` must be `"extracted"` or `"skipped"`, and `skip_reason` is required when skipping.

---

### `finish_bronze_extraction`

Validates that all service chunks have been marked extracted or skipped. Returns an error listing pending chunk IDs if any remain. Transitions the session to `BRONZE_COMPLETE`. Call this before writing Silver, then `submit_inventory_probes`, then `complete_ingest`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_key` | string (required) | Session key |

---

### `complete_ingest`

Completes the ingest session after Silver has been written and `submit_inventory_probes` satisfied (in `answer_complete` mode). Requires `finish_bronze_extraction` first. Removes the session from memory and the `ingest_sessions` SQLite table.

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_key` | string (required) | Session key |

---

### `ingest_url`

Fetches a URL and starts the **same stateful ingest session** as `ingest_file` (not a static header dump). Follow IRON RULES until `complete_ingest`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `url` | string (required) | `http` or `https` URL to fetch |
| `mode` | string | `answer_complete` (default), `routing`, or `audit` |
| `depth` | string | `quick`, `standard` (default), or `thorough` — same presets as `ingest_file` |
| `force` | boolean | Bypass duplicate `content_hash` check and wipe prior namespace |
| `authority` | string | Issuing authority metadata. Defaults to the URL hostname |
| `doc_slug` | string | Short namespace identifier. Defaults to the last URL path segment |

Returns the same stateful session payload as `ingest_file`: `session_key`, numbered service chunks, and inline IRON RULES. Content is not embedded in the response — read chunks via `get_service_chunk` / `get_service_chunks`.

---

## Protocol Notes

- The server speaks JSON-RPC 2.0 over stdio with newline-delimited JSON (one object per line)
- All tool calls return `content: [{type: "text", text: "..."}]` on success
- Errors return `isError: true` with the error message in `content[0].text`
- The server is single-threaded; concurrent tool calls from one session are serialized
