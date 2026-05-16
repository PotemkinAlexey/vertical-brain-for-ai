# Vertical Brain

A personal context lakehouse for AI assistants — hierarchical, layered, transactional, with zero Python runtime dependencies.

Vertical Brain gives your AI a **structured memory** instead of a flat context window. Knowledge is organized into namespace hierarchies, promoted through Bronze → Silver → Gold quality layers, and exposed to models through locked context capsules that prevent cross-domain leakage.

---

## Why

Chat assistants forget between sessions and mix unrelated contexts together. Vector DBs store isolated facts without structure. Vertical Brain takes a different approach:

- **Namespaces** enforce vertical isolation — `WORK/DataArt/Databricks` never bleeds into `PERSONAL/Finance`
- **Layers** distinguish raw notes (Bronze) from canonical facts (Silver) from stable summaries (Gold)
- **Locked context** gives the model a bounded, focused view — exactly what it needs, nothing it doesn't
- **Operations** make writes explicit, validated, and auditable — no silent mutations

---

## Quick Start

```bash
pip install -e .          # no external dependencies
vb ingest "Spark 3.5 dropped support for Python 3.8"
vb map
vb search "Spark Python"
vb context search "Spark compatibility"
```

### MCP (Claude Desktop)

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vertical-brain": {
      "command": "vb",
      "args": ["--data-dir", "/path/to/your/brain", "mcp"]
    }
  }
}
```

With semantic search via Ollama:

```json
{
  "mcpServers": {
    "vertical-brain": {
      "command": "vb",
      "args": [
        "--data-dir", "/path/to/your/brain",
        "mcp",
        "--embedding-url", "http://localhost:11434/v1/embeddings",
        "--embedding-model", "nomic-embed-text"
      ]
    }
  }
}
```

---

## Core Concepts

### Namespace Hierarchy

Every piece of knowledge lives at a path:

```
ROOT
├── WORK
│   ├── DataArt
│   │   ├── Databricks        ← chunks about Databricks at DataArt
│   │   └── MLflow
│   └── SideProject
└── PERSONAL
    └── Finance
```

Paths are slash-separated. Vertical isolation means context opened at `WORK/DataArt/Databricks` includes ancestors (`WORK/DataArt`, `WORK`) but never peers like `PERSONAL/Finance` unless you follow an explicit link.

### Knowledge Layers

| Layer | Purpose | Managed by |
|-------|---------|------------|
| **Bronze** | Raw notes, questions, fragments | You / the model |
| **Silver** | Canonical facts, cleaned, deduplicated | Optimizer compaction |
| **Gold** | Stable summary aspects, stable IDs | `append_gold_aspect` / `GoldBuilder` |

The optimizer deduplicates Bronze chunks and compacts them into Silver summaries. Gold aspects carry stable UUIDs so they survive rewrites without identity drift. Up to 20 Gold aspects per namespace; overflow creates a sibling namespace automatically.

### Operations

All writes are explicit `StorageOperation` objects validated against JSON Schema before execution:

| Operation | What it does |
|-----------|-------------|
| `append_chunk` | Add a new chunk to a namespace |
| `append_gold_aspect` | Add or refresh a semantic label in the Gold layer |
| `create_link` | Create a horizontal link between namespaces |
| `mark_stale` | Retire a chunk |
| `supersede_chunk` | Replace chunks with a newer version |
| `rename_namespace` | Atomically rename a namespace prefix |

### Locked Context

The model never sees raw storage dumps. It receives a **locked context capsule** — a bounded view containing:
- Full chunk content (Gold → Silver → Bronze priority)
- Ancestor Gold summaries for orientation
- Link handles (not expanded content) for horizontal navigation

---

## CLI Reference

All commands share global flags:

```
vb [--data-dir DIR] [--storage-backend sqlite|json] [--model-file FILE] COMMAND
```

| Command | Description |
|---------|-------------|
| `ingest TEXT` | Route and store a piece of text |
| `ask QUESTION` | Route a question and show the locked context |
| `map` | Show the namespace map with Gold summaries |
| `search QUERY` | Lexical full-text search |
| `search --semantic QUERY` | Semantic embedding search |
| `route TEXT` | Show best-matching namespaces by embedding similarity |
| `context search QUERY` | Search + open locked context capsules |
| `context expand LINK_ID` | Expand a link handle into a locked context |
| `session-start` | Print the orientation prompt for a new model session |
| `tree` | Print the raw namespace tree |
| `optimize PATH` | Run duplicate detection and Silver compaction |
| `optimize --plan PATH` | Show compaction plan without applying |
| `doctor` | Run storage integrity checks |
| `operation dry-run FILE` | Validate an operation batch JSON file |
| `operation apply FILE` | Apply an operation batch JSON file |
| `mcp` | Start the MCP stdio server |

### Examples

```bash
# Map the whole brain
vb map

# Map a subtree with depth limit
vb map --path WORK --max-depth 2

# Search within a subtree
vb search --path WORK/DataArt "Databricks"

# Semantic search with Ollama
vb search --semantic \
  --embedding-url http://localhost:11434/v1/embeddings \
  "Python 3.8 compatibility"

# Open locked context capsules for a query
vb context search "MLflow experiment tracking" --path WORK

# Show compaction plan for a namespace
vb optimize --plan WORK/DataArt/Databricks

# Apply a model-emitted operation batch
vb operation apply ops.json

# Run integrity checks
vb doctor
```

---

## Storage Backends

### SQLite (default, production)

Single-file `vertical_brain.sqlite` inside `--data-dir` with WAL mode. Supports FTS5 full-text search, transactional batches, persistent embedding vector cache, and an append-only operation audit log.

**Concurrency:** one writer + N readers via WAL. Use `ThreadLocalSQLiteStoreProxy` for multi-threaded access — it creates one `SQLiteStore` instance per thread.

Use SQLite for durable personal or agent-backed memory. It is the production storage backend for transactional operation batches, OCC, audit history, FTS5 search, and persistent vector cache.

```bash
vb --data-dir ./brain ...
```

### JSON (dev/debug only)

Human-readable files: `nodes.json`, `chunks.json`, `links.json`, `vector_cache.json`, `operation_audit.jsonl`. Good for inspecting and editing state by hand.

JsonStore uses atomic file replacement and rolls back in-process operation batches on exceptions, but it is not safe for multi-process writers and is not a crash-safe database. Do not use it as the production backend.

```bash
vb --data-dir ./brain --storage-backend json ...
```

---

## Semantic Search & Routing

Vertical Brain supports semantic search via any OpenAI-compatible embeddings endpoint. Embedding vectors are cached persistently to avoid recomputation across sessions. Switching models triggers automatic cache invalidation.

```bash
# Semantic search
vb search --semantic \
  --embedding-url http://localhost:11434/v1/embeddings \
  --embedding-model nomic-embed-text \
  "query text"

# Route text to the best matching namespace
vb route \
  --embedding-url http://localhost:11434/v1/embeddings \
  "Databricks cluster autoscaling"
```

---

## Operation Contracts

Models emit operations as JSON. Vertical Brain validates them against JSON Schema before applying:

```json
{
  "operations": [
    {
      "operation": "append_chunk",
      "target_path": "WORK/DataArt/Databricks",
      "chunk": {
        "content": "Delta Lake Z-ordering reduces scan time by clustering related rows.",
        "layer": "silver",
        "content_type": "fact",
        "source": "model",
        "confidence": 0.92
      },
      "reasoning_summary": "Canonicalized user note on Z-ordering performance."
    }
  ],
  "reasoning_summary": "Ingest performance fact."
}
```

```bash
vb operation apply ops.json      # apply
vb operation dry-run ops.json    # validate only
```

See [docs/04_operations_reference.md](docs/04_operations_reference.md) for the full operation schema.

---

## MCP Tools

The MCP server exposes these tools to Claude:

| Tool | Purpose |
|------|---------|
| `session_start` | Orientation prompt at session start (call first) |
| `namespace_map` | Full namespace map as JSON |
| `list_chunks` | List chunks at a path |
| `read_context` | Open a locked context capsule |
| `search` | Lexical FTS search |
| `search_semantic` | Semantic embedding search |
| `context_search` | Search + locked context capsules |
| `context_search_semantic` | Semantic search + locked context capsules |
| `route` | Find best namespaces by embedding similarity |
| `append_chunk` | Write a chunk |
| `append_gold_aspect` | Add/refresh a Gold aspect |
| `create_link` | Create a namespace link |
| `mark_stale` | Retire chunks at a namespace |
| `batch_append` | Write multiple chunks atomically |
| `session_end` | Persist session summary + optional Gold aspect |
| `operations` | Apply a `StorageOperationBatch` JSON object |
| `optimize` | Run optimizer on a subtree |
| `doctor` | Run storage integrity checks |

See [docs/05_mcp_tools.md](docs/05_mcp_tools.md) for full parameter reference.

---

## Development

```bash
pip install -e .
pytest
pytest tests/test_operations.py -v   # run a subset
```

**Zero Python runtime dependencies.** Core, storage, and MCP server use only the Python standard library. The optional HTTP embedding provider (`HttpEmbeddingProvider`) uses `urllib` from stdlib. Optional semantic search requires an external OpenAI-compatible embedding endpoint (e.g. Ollama, OpenAI).

See [ARCHITECTURE.md](ARCHITECTURE.md) for a deep dive into the design.
