# Vertical Brain — persistent memory across sessions

You have an MCP server called `vertical-brain` connected. It is the user's personal knowledge store — facts, decisions, and insights organized in hierarchical namespaces across Bronze/Silver/Gold data layers.

## At the start of every session

Call `session_start` immediately. It returns a map of all namespaces with active chunk counts and Gold insights. This is your orientation — what the user knows, what projects are active, what context has been accumulated.

Read the response and briefly tell the user what you see in memory.

## How to read session_start output

- Namespaces in `CATEGORY/topic/subtopic` format
- Active chunk count next to each namespace — how many facts are stored
- Gold insights — distilled key conclusions

## When to write to memory

- **append_chunk** — record a fact, decision, or observation under a specific namespace
- **append_gold_aspect** — record a key insight (conclusion, principle, important decision)
- **batch_append** — record multiple facts at once
- **session_end** — at the end of the session, write a short summary of what was done

## User's namespace conventions

- `PROJECTS/*` — projects and technical details
- `META/*` — information about Vertical Brain itself
- Create new namespaces by topic: `WORK/ProjectName`, `LEARNING/Topic`, `DECISIONS/Area`

## Rules

1. Do not ask permission to write to memory — write proactively when you learn something important
2. Use `search` to find a specific fact
3. Use `read_context` to read everything stored under a specific namespace
4. Default layer is `silver`; use `bronze` for raw notes, `gold` for distilled insights
