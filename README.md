# Vertical Brain MVP

**Vertical Brain** is a personal context lakehouse.

It stores knowledge in isolated vertical namespaces, routes every new input into the deepest relevant partition, prevents cross-domain context leakage, and continuously compacts raw notes into stable knowledge summaries.

## Core idea

Vertical Brain is not a generic chatbot memory.

It is a controlled knowledge storage engine with:

- strict namespace isolation
- vertical context routing
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

Initial architecture scaffold for Codex.
