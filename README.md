# Vertical Brain MVP

**Vertical Brain** is a personal context lakehouse.

It stores knowledge in isolated vertical namespaces, accepts strict storage operations from models, prevents cross-domain context leakage, and continuously compacts raw notes into stable knowledge summaries.

## Core idea

Vertical Brain is not a generic chatbot memory and it is not the model.

It is a controlled knowledge storage engine with:

- strict namespace isolation
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

Use SQLite when you need transactional local storage:

```bash
vb --storage-backend sqlite ingest "Models should write through validated operation batches."
```

Use a custom storage model when changing the format contract:

```bash
vb --model-file data/namespaces/model.json tree
```

`model.json` describes storage and routing contracts only. It does not contain domain routing rules; route decisions are strict JSON returned by the configured model provider. For local tests, `--llm-response-file` can inject a canned provider response.

Inspect router decisions as strict JSON:

```bash
vb ingest --route-json "Databricks Auto Loader schema evolution"
vb ask --route-json "How does Databricks schema evolution work?"
```

If router confidence is below the namespace model threshold, the CLI asks for clarification instead of writing low-quality knowledge into the store.

Models can also emit first-class storage operations directly. `dry-run` validates
the operation or batch without mutating storage; `apply` prevalidates the full
batch before any write:

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

## Project status

MVP core implemented:

- CLI ingest / ask / tree / optimize
- local JSON storage
- local SQLite storage with transaction support for operation batches
- root namespace bootstrap from `data/namespaces/root.json`
- LLM-driven routing through strict JSON contracts from `data/namespaces/model.json`
- first-class `StorageOperation` execution
- `StorageOperationBatch` optimization for many variants into one canonical chunk
- operation validation, dry-run, and batch preflight before writes
- route decision JSON serialization
- locked context capsules with budget and horizontal link handles
- exact duplicate stale marking
- namespace-bounded compaction into Silver chunks
- deterministic Gold summary artifacts
- context-bound MVP answerer
