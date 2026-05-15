# MVP CLI Workflow

## Commands

The default backend is JSON. SQLite can be selected for transactional local
storage:

```bash
vb --storage-backend sqlite ingest "New information"
```

### Ingest

```bash
vb ingest "For Delta streaming sink schema evolution use mergeSchema=true"
```

Expected output:

```text
Target path:
WORK/DataArt/Databricks/Certification/StructuredStreaming/SchemaEvolution

Action:
append_and_optimize

Peer links:
- WORK/DataArt/Databricks/Certification/AutoLoader/SchemaEvolution

Status:
written and indexed
```

### Ask

```bash
vb ask "How do I enable schema evolution for a streaming Delta sink?"
```

Expected behavior:

```text
1. Classify target vertical.
2. Lock unrelated branches.
3. Retrieve target node, ancestors, and approved peer links.
4. Answer only from allowed context.
```

### Search

```bash
vb search --path WORK/DataArt "mergeSchema"
vb --storage-backend sqlite search "schema evolution"
```

Expected behavior:

```text
1. Search chunks and Gold summaries.
2. Limit results to the requested branch when --path is set.
3. Exclude stale and superseded chunks unless --include-stale is set.
4. Return ranked candidates with path, source, layer, type, and snippet.
5. Use SQLite FTS when SQLite backend is selected; use lexical fallback for JSON.
```

### Tree

```bash
vb tree
```

Expected output:

```text
WORK
  DataArt
    Databricks
      Certification
    FXDB
PERSONAL
TRADING
```

### Optimize

```bash
vb optimize WORK/DataArt/Databricks/Certification
```

Expected behavior:

```text
1. Load chunks under selected branch.
2. Group related chunks.
3. Mark stale or superseded chunks.
4. Produce updated Silver blocks.
5. Update Gold summary.
```

### Operation

```bash
vb operation dry-run operation.json
vb operation apply operation.json
```

Expected behavior:

```text
1. Read a StorageOperation or StorageOperationBatch JSON object.
2. Validate JSON shape against the operation schema in model.json.
3. Validate protocol invariants against current storage state.
4. For dry-run, return validation and planned result without writing.
5. For apply, validate the full batch before the first write.
6. Return strict JSON with operation results and validation status.
```
