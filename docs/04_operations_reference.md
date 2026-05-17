# Operations Reference

A `StorageOperation` is a validated write intent. The executor validates every operation against the storage model JSON Schema before any I/O, then applies the whole batch transactionally.

---

## Batch Format

Send a `StorageOperationBatch` (an object with an `operations` array):

```json
{
  "operations": [ ... ],
  "reasoning_summary": "Human-readable explanation of why this batch was emitted.",
  "branch_path": "WORK/DataArt",
  "start_version": 7
}
```

| Field | Type | Description |
|-------|------|-------------|
| `operations` | array | **required** List of `StorageOperation` objects |
| `reasoning_summary` | string | Why this batch was emitted (goes to audit log) |
| `branch_path` | string | OCC guard: namespace whose version must match `start_version` |
| `start_version` | integer | OCC guard: expected current version of `branch_path` |

`branch_path` + `start_version` together enable **optimistic concurrency control**. If `branch_path` is provided and the node's current version differs from `start_version`, the entire batch is rejected with `OptimisticLockException`.

---

## Operations

### `append_chunk`

Add a new chunk to a namespace. Creates the node chain if nodes do not exist.

```json
{
  "operation": "append_chunk",
  "target_path": "WORK/DataArt/Databricks",
  "chunk": {
    "content": "Delta Lake Z-ordering clusters rows by selected columns to reduce scan range.",
    "layer": "silver",
    "content_type": "fact",
    "source": "model",
    "confidence": 0.92
  },
  "confidence": 0.92,
  "reasoning_summary": "Canonicalized performance note."
}
```

**`chunk` fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `content` | string | required | The text |
| `layer` | `bronze`\|`silver`\|`gold` | `bronze` | Knowledge quality layer |
| `content_type` | enum | `note` | `fact`, `decision`, `question`, `note`, `code`, `artifact`, `correction` |
| `source` | string | `""` | Who produced this (`user`, `model`, `optimizer:namespace_compaction`, …) |
| `confidence` | float [0,1] | `1.0` | Routing/quality signal |
| `lineage` | string[] | `[]` | IDs of source chunks this was distilled from |

**Bronze dedup enforcement (layer `bronze` only):**

- **Hard block — exact duplicate.** If an active Bronze chunk with byte-for-byte identical content already exists at `target_path`, the operation is rejected with a `ValueError` before any I/O. The error names the existing chunk ID and directs you to call `mark_stale` first.
- **Soft warning — similar content.** If the write succeeds but the FTS search finds lexically similar active Bronze chunks at `target_path` (score ≥ 8), `OperationResult.similar_bronze` is populated with up to 3 matches. Review and `mark_stale` any that are superseded by the new chunk.

```json
{
  "operation": "append_chunk",
  "status": "applied",
  "chunk_id": "new-id...",
  "similar_bronze": [
    { "chunk_id": "old-id...", "snippet": "Databricks uses Delta Lake...", "score": 10 }
  ]
}
```

**`layer=silver` guard:** if an active Silver chunk already exists at `target_path`, `append_chunk(layer=silver)` is rejected with a `ValueError`. Use `append_chunk(layer=silver)` only to create the **first** Silver summary at a new namespace. All subsequent Silver updates must use `update_silver`.

Gold chunks are exempt from Bronze dedup checks. Silver is exempt from dedup but subject to the first-Silver guard above.

---

### `update_silver`

Replace the active Silver summary for a namespace atomically. Supersedes the old Silver chunk and writes a new one with full lineage tracking. Enforces OCC.

> **First Silver:** if no active Silver exists yet, create it with `append_chunk(layer="silver")`. After that, always use `update_silver`.

```json
{
  "operation": "update_silver",
  "target_path": "WORK/Topic",
  "current_silver_id": "old-silver-id",
  "source_chunk_ids": ["new-bronze-id-1", "new-bronze-id-2"],
  "chunk": {
    "content": "Updated complete Silver summary incorporating new facts.",
    "layer": "silver",
    "content_type": "note",
    "source": "model",
    "confidence": 0.9
  },
  "reasoning_summary": "Incorporated two new Bronze facts into Silver."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `target_path` | string | **required** Namespace to update |
| `current_silver_id` | string | **required** ID of the active Silver chunk you read before synthesizing |
| `chunk` | object | **required** New Silver content; `layer` must be `"silver"` |
| `source_chunk_ids` | string[] | Optional Bronze chunk IDs incorporated into this Silver (stored in lineage) |

**Validation rejects:**
- `current_silver_id` missing or empty
- `chunk` missing or `chunk.content` empty
- `chunk.layer` ≠ `"silver"`
- Duplicate or empty strings in `source_chunk_ids`

**Runtime rejects (apply-time, not just validation):**
- `current_silver_id` chunk not found
- `current_silver_id` chunk is not active (already superseded/stale)
- `current_silver_id` chunk is not `layer=silver`
- `current_silver_id` chunk belongs to a different namespace
- Any `source_chunk_ids` entry is `layer=gold` (Gold cannot be source evidence for Silver)

**Result:** old Silver gets `status=superseded` + `valid_to=now()`. New Silver gets `lineage=[current_silver_id] + source_chunk_ids` and `supersedes=[current_silver_id]`. Exactly one active Silver remains after the operation.

---

### `append_gold_aspect`

> **Prerequisite:** An active Silver chunk must exist at `target_path`. If none is found, the operation is rejected with a `ValueError` instructing you to call `update_silver` first.

Add or refresh a short semantic routing tag in the Gold index of a namespace.

- Exact-text duplicates refresh `updated_at` without creating a new aspect
- Each aspect gets a stable UUID that persists across rewrites
- Max 20 aspects per node; overflow creates a sibling namespace (`{path}_2`, `{path}_3`, …)
- Aspects are embedded individually by `EmbeddingRouter`; use precise search tags, not summaries
- Aspects longer than 150 characters are written but return `aspect_too_long: true`

```json
{
  "operation": "append_gold_aspect",
  "target_path": "WORK/DataArt/Databricks",
  "gold_aspect": "Delta Lake Z-ordering lookup",
  "reasoning_summary": "Stable Gold routing tag for Z-ordering knowledge."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `target_path` | string | **required** Namespace to update |
| `gold_aspect` | string | **required** Short search tag; target ~30–100 characters |

**Result warnings:**

```json
{
  "operation": "append_gold_aspect",
  "status": "applied",
  "aspect_too_long": true
}
```

If `aspect_too_long` is true, split the aspect into smaller, semantically distinct search tags and append them separately.

> **MCP vs StorageOperation naming:** The MCP `append_gold_aspect` tool accepts the argument as `aspect`, but the underlying `StorageOperation` field is `gold_aspect`. When building operation batches directly (e.g. via the `operations` MCP tool or `vb operation apply`), use `gold_aspect`.

---

### `create_link`

Create a horizontal link between two namespaces. Links appear as handles in locked context capsules.

The `target_path` on the operation is the **source** namespace. Each entry in `links` describes a connection to a target namespace.

```json
{
  "operation": "create_link",
  "target_path": "WORK/DataArt/Databricks",
  "links": [
    {
      "target_path": "WORK/SideProject/DataPipeline",
      "link_type": "reference",
      "reason": "Pipeline uses Databricks Delta Lake as the target store."
    }
  ],
  "reasoning_summary": "Cross-domain reference: Databricks → SideProject pipeline."
}
```

Multiple links can be created in one operation by including multiple entries in the `links` array.

**`links` item fields:**

| Field | Type | Description |
|-------|------|-------------|
| `target_path` | string | **required** Namespace the link points to |
| `link_type` | string | **required** e.g. `peer`, `reference`, `derived_from`, `depends_on` |
| `reason` | string | **required** Human-readable explanation |

---

### `mark_stale`

Mark one or more chunks as stale. Sets `status = "stale"` and `valid_to = now()`.

```json
{
  "operation": "mark_stale",
  "target_path": "WORK/DataArt/Databricks",
  "chunk_ids": ["550e8400-e29b-41d4-a716-446655440000"],
  "reasoning_summary": "Fact superseded by updated Delta Lake documentation."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `target_path` | string | **required** Namespace |
| `chunk_ids` | string[] | **required** IDs of chunks to mark stale |

---

### `supersede_chunk`

Mark chunks as superseded (typically used by the optimizer after compaction). Sets `status = "superseded"` and `valid_to = now()`.

```json
{
  "operation": "supersede_chunk",
  "target_path": "WORK/DataArt/Databricks",
  "chunk_ids": ["id-1", "id-2"],
  "reasoning_summary": "Original variants superseded by canonical Silver compaction."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `target_path` | string | **required** Namespace |
| `chunk_ids` | string[] | **required** IDs of chunks to supersede |

---

### `rename_namespace`

Atomically rename a namespace prefix. Renames all chunks, links, and nodes under the prefix. Validates that:
- The old prefix exists
- The new prefix does not already exist
- The new prefix is not a descendant of the old prefix (would create a cycle)

```json
{
  "operation": "rename_namespace",
  "target_path": "OLD/Project",
  "new_path": "NEW/Project",
  "reasoning_summary": "Reorganized domain hierarchy."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `target_path` | string | **required** Current namespace prefix |
| `new_path` | string | **required** New namespace prefix |

---

## Applying Batches

### Via CLI

```bash
# Validate only
vb operation dry-run ops.json

# Apply
vb operation apply ops.json
```

### Via MCP

Call the `operations` tool with the batch as the `payload` parameter.

### Response Format

`OperationBatchResult` JSON — one entry in `results` per applied operation:

```json
{
  "status": "applied",
  "results": [
    {
      "operation": "append_chunk",
      "target_path": "WORK/DataArt/Databricks",
      "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
      "link_ids": [],
      "status": "applied",
      "validation": {"valid": true, "issues": []},
      "overflow_path": null
    }
  ],
  "validation": {"valid": true, "issues": []}
}
```

On validation failure (e.g. from `dry-run`):

```json
{
  "status": "invalid",
  "results": [],
  "validation": {
    "valid": false,
    "issues": [
      {
        "path": "operations[0].chunk.layer",
        "message": "must be one of ['bronze', 'silver', 'gold']",
        "severity": "error"
      }
    ]
  }
}
```

---

## Staging Buffer

When the router has low confidence (or the model emits `ask_clarification`), content is staged to `STAGING/Unclassified` with `confidence=0.0`. Use `vb doctor` to identify staging items that are older than 3 days. Reclassification is intentional and must be done explicitly — write a `StorageOperationBatch` that moves or supersedes the staged chunk at its correct target path. `vb optimize STAGING` compacts and deduplicates within the staging namespace but does not automatically reclassify staged items.
