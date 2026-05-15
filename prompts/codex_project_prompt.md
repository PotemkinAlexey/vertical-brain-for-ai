# Codex Prompt: Build Vertical Brain MVP v0.1

You are building an MVP Python project called **Vertical Brain**.

Vertical Brain is a personal context lakehouse. It stores knowledge in isolated vertical namespaces and prevents LLM context pollution by routing every input and query into the correct vertical path.

## Goal

Create a local CLI-first MVP.

The MVP must support:

```bash
vb ingest "some new knowledge"
vb ask "some question"
vb tree
vb optimize WORK/DataArt/Databricks
```

## Required architecture

Use Python.

Start simple. Do not over-engineer.

Use local JSON storage first. SQLite can be added later.

Recommended structure:

```text
src/vertical_brain/
  core/
    models.py
    router.py
    optimizer.py
    context_lock.py
  storage/
    json_store.py
  llm/
    prompt_templates.py
    mock_llm.py
  cli/
    main.py
```

## Core concepts

### Node

A namespace path:

```text
WORK/DataArt/Databricks/Certification
```

### Chunk

A knowledge item stored under a node.

Chunk fields:

```text
id
node_path
layer
content
content_type
status
source
confidence
created_at
updated_at
```

### Link

Explicit relationship between chunks or nodes.

### RouteDecision

Structured JSON returned by the router.

## Implementation requirements

1. Implement dataclasses or Pydantic models for:
   - Node
   - Chunk
   - Link
   - RouteDecision

2. Implement local JSON storage:
   - create nodes
   - save chunks
   - list tree
   - find chunks by path
   - find ancestors
   - find peer links

3. Implement a mock router first:
   - If input contains "Databricks", route to `WORK/DataArt/Databricks`
   - If input contains "dbt", route to `WORK/Stack/dbt`
   - If input contains "trading", route to `TRADING`
   - Otherwise route to `INBOX/Unclassified`

4. Implement CLI with Typer or argparse:
   - `vb ingest`
   - `vb ask`
   - `vb tree`
   - `vb optimize`

5. Implement context locking:
   - For `ask`, retrieve only:
     - target node chunks
     - ancestor node summaries
     - explicitly linked peer nodes
   - Do not retrieve globally.

6. Implement simple optimize:
   - Group chunks by node path.
   - Create or update a Gold summary placeholder.
   - Mark duplicate chunks as stale only if exact duplicate for now.
   - Leave semantic optimize as TODO.

7. Add tests for:
   - routing
   - ingest
   - tree listing
   - context lock
   - duplicate stale detection

## Important constraints

- Do not build UI yet.
- Do not add vector DB yet.
- Do not add background workers yet.
- Do not add LangChain/LlamaIndex yet.
- Keep the storage engine explicit and understandable.
- Every routing result must be inspectable.

## Output expected

Create the full project skeleton with working CLI, minimal tests, and README instructions.

Prioritize clarity over complexity.
