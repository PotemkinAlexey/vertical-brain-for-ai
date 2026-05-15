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
```

## Local usage

Run directly from the repository:

```bash
PYTHONPATH=src python -m vertical_brain.cli.main ingest "Databricks Auto Loader uses Spark Structured Streaming."
PYTHONPATH=src python -m vertical_brain.cli.main ask "How does Databricks schema evolution work?"
PYTHONPATH=src python -m vertical_brain.cli.main tree
PYTHONPATH=src python -m vertical_brain.cli.main optimize WORK/DataArt/Databricks
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
CLASSIFY QUERY → LOCK TARGET VERTICAL → RETRIEVE ONLY ALLOWED CONTEXT → ANSWER
```

## Project status

MVP core implemented:

- CLI ingest / ask / tree / optimize
- local JSON storage
- root namespace bootstrap from `data/namespaces/root.json`
- LLM-driven routing through strict JSON contracts from `data/namespaces/model.json`
- first-class `StorageOperation` execution
- route decision JSON serialization
- context locking with ancestors, target node, and explicit peer links
- exact duplicate stale marking
- namespace-bounded compaction into Silver chunks
- deterministic Gold summary artifacts
- context-bound MVP answerer
