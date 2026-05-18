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

Vectors are cached persistently — embeddings are computed once and reused across sessions. Switching models triggers automatic cache invalidation.

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

The MCP server exposes these tools to Claude:

| Tool | Purpose |
|------|---------|
| `session_start` | Orientation prompt at session start — **call first** |
| `read_context` | Open a locked context capsule for a namespace |
| `search` | Lexical FTS search |
| `search_semantic` | Semantic embedding search |
| `context_search` | Search + locked context |
| `route` | Find best namespaces by embedding similarity |
| `append_chunk` | Write a Bronze chunk (or first Silver with `layer=silver`) |
| `update_silver` | Atomically replace the active Silver summary (OCC-protected) |
| `append_gold_aspect` | Add/refresh a short Gold routing tag (requires active Silver) |
| `create_link` | Create a horizontal link between namespaces |
| `mark_stale` | Retire a chunk |
| `batch_append` | Write multiple chunks atomically |
| `session_end` | Persist session notes + Silver summary + optional Gold aspect |
| `optimize` | Run dedup + compaction on a namespace subtree |
| `vacuum` | Preview/apply cleanup of inactive chunks |
| `doctor` | Storage integrity checks |

See [docs/05_mcp_tools.md](docs/05_mcp_tools.md) for full parameter reference.

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
pytest                              # full suite (~469 tests)
pytest tests/test_operations.py -v  # single file
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
