# MVP CLI Workflow

## Commands

The default backend is SQLite. The JSON backend can be selected for human-readable local storage:

```bash
vb --storage-backend json ingest "New information"
```

### Map

```bash
vb map
vb map --path WORK/DataArt --max-depth 2
vb map --json --summary-chars 160
```

Expected behavior:

```text
1. Show namespace paths as a model-facing map.
2. Include child paths, chunk counts, subtree counts, Gold summaries, and link handles.
3. Do not expose raw chunk content.
4. Limit map scope with --path and relative depth with --max-depth.
5. Return strict JSON when --json is set.
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

### Context Search

```bash
vb context search --path WORK/DataArt "mergeSchema"
vb context search --json --context-limit 2 --items-per-context 4 "schema evolution"
```

Expected behavior:

```text
1. Search for candidate namespace paths.
2. Return candidate handles without using raw search snippets as model context.
3. Dedupe candidate paths.
4. Open bounded locked context capsules for selected paths.
5. Keep horizontal links as handles unless link expansion is explicitly requested.
```

### Context Expand

```bash
vb context expand <link_id> --from-path WORK/DataArt/Databricks
vb context expand <link_id> --json --items-per-context 4
```

Expected behavior:

```text
1. Resolve the horizontal link by id.
2. If --from-path is provided, verify the link is connected to that path.
3. Open the other side as a separate locked context.
4. Keep links in the expanded context as handles unless link expansion is explicitly requested.
5. Return strict JSON when --json is set.
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
