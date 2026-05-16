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

---

### `append_gold_aspect`

Add or refresh a semantic label in the Gold summary of a namespace.

- Exact-text duplicates refresh `updated_at` without creating a new aspect
- Each aspect gets a stable UUID that persists across rewrites
- Max 20 aspects per node; overflow creates a sibling namespace (`{path}_2`, `{path}_3`, …)

```json
{
  "operation": "append_gold_aspect",
  "target_path": "WORK/DataArt/Databricks",
  "aspect": "Delta Lake Z-ordering reduces scan range by clustering rows on selected columns.",
  "reasoning_summary": "Stable Gold label for Z-ordering knowledge."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `target_path` | string | **required** Namespace to update |
| `aspect` | string | **required** The semantic label text |

---

### `create_link`

Create a horizontal link between two namespaces. Links appear as handles in locked context capsules.

```json
{
  "operation": "create_link",
  "target_path": "WORK/DataArt/Databricks",
  "link": {
    "source_path": "WORK/DataArt/Databricks",
    "target_path": "WORK/SideProject/DataPipeline",
    "link_type": "reference",
    "reason": "Pipeline uses Databricks Delta Lake as the target store."
  },
  "reasoning_summary": "Cross-domain reference: Databricks → SideProject pipeline."
}
```

**`link` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `source_path` | string | **required** |
| `target_path` | string | **required** |
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

```json
{
  "status": "ok",
  "applied": 3,
  "validation": {
    "valid": true,
    "errors": []
  }
}
```

On validation failure with `dry-run`:

```json
{
  "status": "invalid",
  "applied": 0,
  "validation": {
    "valid": false,
    "errors": ["operations[0].chunk.layer: must be one of ['bronze', 'silver', 'gold']"]
  }
}
```

---

## Staging Buffer

When the router has low confidence (or the model emits `ask_clarification`), content is staged to `STAGING/Unclassified` with `confidence=0.0`. Run `vb optimize STAGING` to batch-process staged items during a dedicated triage session, or use `vb doctor` to see how many items have been staged for more than 3 days.
