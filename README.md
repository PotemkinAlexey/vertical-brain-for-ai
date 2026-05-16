# Vertical Brain MVP

**Vertical Brain** is a personal context lakehouse.

It stores knowledge in isolated vertical namespaces, accepts strict storage operations from models, prevents cross-domain context leakage, and continuously compacts raw notes into stable knowledge summaries.

## Core idea

Vertical Brain is not a generic chatbot memory and it is not the model.

It is a controlled knowledge storage engine with:

- strict namespace isolation
- fast namespace-scoped search
- vertical context storage operations
- Bronze / Silver / Gold knowledge layers
- peer-links between controlled sibling nodes
- stale context detection
- read-time context locking
- optimize / compaction process

## MVP goal

Build a CLI-first prototype that can:

```bash
vb ingest "New information"
vb ask "Question"
vb map
vb search "Question or keyword"
vb context search "Question or keyword"
vb tree
vb optimize WORK/DataArt/Databricks
vb operation dry-run operation.json
vb operation apply operation.json
```

## Local usage

Run directly from the repository:

```bash
PYTHONPATH=src python -m vertical_brain.cli.main ingest "Databricks Auto Loader uses Spark Structured Streaming."
PYTHONPATH=src python -m vertical_brain.cli.main ask "How does Databricks schema evolution work?"
PYTHONPATH=src python -m vertical_brain.cli.main map --path WORK/DataArt
PYTHONPATH=src python -m vertical_brain.cli.main search --path WORK/DataArt "mergeSchema"
PYTHONPATH=src python -m vertical_brain.cli.main context search --path WORK/DataArt "mergeSchema"
PYTHONPATH=src python -m vertical_brain.cli.main tree
PYTHONPATH=src python -m vertical_brain.cli.main optimize WORK/DataArt/Databricks
PYTHONPATH=src python -m vertical_brain.cli.main operation dry-run operation.json
```

Or install the CLI entrypoint:

```bash
python -m pip install -e .
vb ingest "Databricks Delta schema evolution uses mergeSchema."
vb ask "How does Databricks schema evolution work?"
```

The current local runner uses a mock provider. Without a real model provider or `--llm-response-file`, routing returns a clarification response instead of guessing.

Use an isolated storage directory when experimenting:

```bash
vb --data-dir .vb-dev-data ingest "dbt staging models are ephemeral in this project."
```

## Storage backends

SQLite is the **default** backend. It provides transactional operation
batches, an FTS search index, and an append-only operation audit log:

```bash
vb ingest "Models should write through validated operation batches."
```

JSON storage is available for development and debugging — it keeps every
record in human-readable `nodes.json` / `chunks.json` / `links.json` files
plus an `operation_audit.jsonl` audit log:

```bash
vb --storage-backend json ingest "Inspect raw records on disk while debugging."
```

## Quick start

```bash
# 1. Orient: print all namespaces, Gold summaries, counts, and link handles
vb session-start

# 2. Retrieve: search returns candidate handles, then bounded locked contexts
vb search "mergeSchema"
vb context search "schema evolution"

# 3. Check integrity: run the doctor for orphan links, duplicates, bad fields
vb doctor
vb doctor --json

# 4. Compact: preview a compaction plan before applying it
vb optimize WORK/DataArt/Databricks --plan
vb optimize WORK/DataArt/Databricks
```

## MCP integration

Vertical Brain ships an MCP stdio server (`vb mcp`). Add it to Claude
Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vertical-brain": {
      "command": "vb",
      "args": ["--data-dir", "/absolute/path/to/brain-data", "mcp"]
    }
  }
}
```

Use a custom storage model when changing the format contract:

```bash
vb --model-file data/namespaces/model.json tree
```

`model.json` describes storage, routing, and operation JSON contracts only. It
does not contain domain routing rules; route decisions are strict JSON returned
by the configured model provider. For local tests, `--llm-response-file` can
inject a canned provider response.

Inspect router decisions as strict JSON:

```bash
vb ingest --route-json "Databricks Auto Loader schema evolution"
vb ask --route-json "How does Databricks schema evolution work?"
```

If router confidence is below the namespace model threshold, the CLI asks for clarification instead of writing low-quality knowledge into the store.

Models can also emit first-class storage operations directly. The CLI validates
the incoming JSON against the operation schema from `model.json`, then runs
semantic validation against storage state. `dry-run` validates the operation or
batch without mutating storage; `apply` prevalidates the full batch before any
write:

```json
{
  "operations": [
    {
      "operation": "append_chunk",
      "target_path": "WORK/VerticalBrain/Protocol",
      "chunk": {
        "content": "Models write context through validated storage operations.",
        "layer": "silver",
        "content_type": "fact"
      }
    }
  ],
  "reasoning_summary": "Persist protocol knowledge."
}
```

## Tests

```bash
python -m pytest -q
```

## MVP scope

Version `0.1` should use simple local storage first:

- JSON or SQLite for metadata and chunks
- Markdown files for Gold summaries
- Optional vector index later
- LLM router returning strict JSON
- No UI in the first version

## Key design rule

Do not retrieve from the whole knowledge base.

Always:

```text
MAP FIRST → LOCK TARGET VERTICAL → READ BOUNDED CONTEXT → EXPAND LINKS ONLY ON REQUEST → ANSWER
```

Horizontal links are handles by default. Their target content is not included in
the prompt unless a caller explicitly requests link expansion.

Use `vb map` for the first model-facing read: it exposes namespace structure,
counts, Gold summaries, and link handles without raw chunk content. Use
`vb search` to inspect ranked matches. Use `vb context search` for model-facing
retrieval: search returns candidate handles, then Vertical Brain opens bounded
locked context capsules for selected paths. Raw search snippets are not treated
as model context.

## Project status

MVP core implemented:

- CLI ingest / ask / map / search / context search / tree / optimize
- local JSON storage
- local SQLite storage with transaction support for operation batches
- SQLite FTS search index with JSON lexical fallback
- model-facing namespace map without raw chunk content
- model-facing search-to-locked-context session
- root namespace bootstrap from `data/namespaces/root.json`
- LLM-driven routing through strict JSON contracts from `data/namespaces/model.json`
- first-class `StorageOperation` execution
- `StorageOperationBatch` optimization for many variants into one canonical chunk
- JSON Schema validation for model-emitted operation payloads
- operation validation, dry-run, and batch preflight before writes
- route decision JSON serialization
- locked context capsules with budget and horizontal link handles
- exact duplicate stale marking
- namespace-bounded compaction into Silver chunks
- deterministic Gold summary artifacts
- context-bound MVP answerer
