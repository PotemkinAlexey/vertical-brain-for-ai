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

## Installation

**Requirements:** Python 3.11+. No external runtime dependencies.

```bash
git clone https://github.com/PotemkinAlexey/vertical-brain-for-ai.git
cd vertical-brain-for-ai
pip install -e .
```

Verify:

```bash
vb --help
vb doctor
```

Your data directory defaults to `./data`. Set a persistent location:

```bash
vb --data-dir ~/.brain doctor
```

Or set it once via environment variable (used as the default for `--data-dir`):

```bash
export VB_DATA_DIR=~/.brain
vb doctor
```

---

## Connect to Claude

### Claude Desktop

Find your config file:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Add the MCP server:

```json
{
  "mcpServers": {
    "vertical-brain": {
      "command": "vb",
      "args": ["--data-dir", "/Users/you/.brain", "mcp"]
    }
  }
}
```

Restart Claude Desktop. You should see `vertical-brain` in the tools list.

### Claude Code

Add to your project's `.claude/settings.json`:

```json
{
  "mcpServers": {
    "vertical-brain": {
      "command": "vb",
      "args": ["--data-dir", "/Users/you/.brain", "mcp"]
    }
  }
}
```

Or run directly:

```bash
vb --data-dir ~/.brain mcp
```

### Cursor

Add to `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "vertical-brain": {
      "command": "vb",
      "args": ["--data-dir", "/Users/you/.brain", "mcp"]
    }
  }
}
```

Reload the window. Vertical Brain will appear in Cursor's MCP tools panel.

### OpenAI Codex CLI

Add to `~/.codex/config.json`:

```json
{
  "mcpServers": {
    "vertical-brain": {
      "command": "vb",
      "args": ["--data-dir", "/Users/you/.brain", "mcp"]
    }
  }
}
```

### With semantic search (Ollama)

Install [Ollama](https://ollama.ai), pull an embedding model, then point Vertical Brain at it:

```json
{
  "mcpServers": {
    "vertical-brain": {
      "command": "vb",
      "args": [
        "--data-dir", "/Users/you/.brain",
        "mcp",
        "--embedding-url", "http://localhost:11434/v1/embeddings",
        "--embedding-model", "nomic-embed-text"
      ]
    }
  }
}
```

Vectors are cached persistently — each chunk is embedded once and reused across sessions. The cache is keyed by `(content_hash, model_name)`.

Pick the embedding model before you index and keep it fixed. If you change `--embedding-model` later, semantic search fails fast with `IncompatibleEmbeddingModelError` — the cached vectors belong to the old model and are not comparable to the new one. Re-indexing under the new model (purge old vectors, re-embed every active chunk) is done programmatically via `EmbeddingSearch.trigger_reindexing(new_provider)`; there is not yet a CLI or MCP command for it.

---

## How it works

### Namespace hierarchy

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

Context opened at `WORK/DataArt/Databricks` includes ancestors (`WORK/DataArt`, `WORK`) but never peers like `PERSONAL/Finance` unless you follow an explicit link.

### Knowledge layers

| Layer | Purpose | How it's written |
|-------|---------|-----------------|
| **Bronze** | Raw notes, questions, fragments | `append_chunk` (default) |
| **Silver** | Living summary — one per namespace, kept current | `append_chunk(layer=silver)` (first time) → `update_silver` (all subsequent) |
| **Gold** | Search index — short stable routing aspects | `append_gold_aspect` (requires active Silver) |

Silver is a living document: the agent creates the first summary, then keeps it current after each new Bronze write. Gold aspects are short search tags embedded individually by the router and cached as one persistent `vector_cache` row per aspect/model; they carry stable UUIDs so they survive rewrites. Up to 20 Gold aspects per namespace; overflow creates a sibling namespace automatically.

### MCP tools

The MCP server exposes **27 tools** to agents. Call `session_start` first, then use the groups below. Full parameters: [docs/05_mcp_tools.md](docs/05_mcp_tools.md).

| Group | Tools |
|-------|--------|
| **Orientation** | `session_start`, `namespace_map`, `list_chunks`, `read_context` |
| **Search** | `search`, `search_semantic`, `context_search`, `context_search_semantic`, `route` |
| **Write** | `append_chunk`, `update_silver`, `append_gold_aspect`, `batch_append`, `create_link`, `mark_stale`, `session_end` |
| **Ingest** | `ingest_file`, `ingest_url`, `get_service_chunk`, `get_service_chunks`, `mark_service_chunk`, `finish_bronze_extraction`, `submit_inventory_probes`, `complete_ingest` |
| **Maintenance** | `optimize`, `vacuum`, `doctor`, `operations` |

File ingest uses stateful sessions with server-enforced **IRON RULES** (`mode=answer_complete` by default). See [AGENTS.md](AGENTS.md).

---

## Storage backends

### SQLite (default, production)

Single-file `vertical_brain.sqlite` with WAL mode. Supports FTS5 full-text search, transactional batches, persistent embedding vector cache, and an append-only audit log.

```bash
vb --data-dir ~/.brain ...
```

### JSON (dev/debug only)

Human-readable files — useful for inspecting state by hand. Not safe for production.

```bash
vb --data-dir ./brain --storage-backend json ...
```

---

## Development

```bash
pip install -e .
.venv/bin/python -m pytest                              # full suite (530 tests)
.venv/bin/python -m pytest tests/test_operations.py -v  # single file
```

**Zero Python runtime dependencies.** Core, storage, and MCP server use only the Python standard library. Optional semantic search requires an external OpenAI-compatible embedding endpoint (e.g. Ollama, OpenAI API).

---

## Docs

- [Architecture manifesto](docs/01_architecture_manifesto.md)
- [Operations reference](docs/04_operations_reference.md)
- [MCP tools reference](docs/05_mcp_tools.md)
- [Agent usage guide](AGENTS.md)
- [Production runbook](docs/06_production_runbook.md)

---

## Advanced: manual CLI operations

For scripting, debugging, or batch imports. Most users will never need this.

```bash
# Map the brain
vb map
vb map --path WORK --max-depth 2

# Search
vb search --path WORK/DataArt "Databricks"
vb search --semantic --embedding-url http://localhost:11434/v1/embeddings "Python 3.8"

# Inspect
vb context search "MLflow experiment tracking" --path WORK
vb tree

# Maintenance
vb optimize WORK/DataArt/Databricks
vb optimize --plan WORK/DataArt/Databricks   # dry-run
vb doctor
vb vacuum --retention-hours 168
vb vacuum --apply --backup ./backups/before-vacuum.sqlite

# SQLite ops
vb backup ./backups/brain.sqlite
vb checkpoint --mode truncate

# Apply a JSON operation batch
vb operation apply ops.json
vb operation dry-run ops.json
```

Full CLI reference: `vb --help`, `vb COMMAND --help`.
