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

## Storage maintenance

- **optimize** with `path` — preferred form: dedup, Silver compaction, decay, then global link discovery, scoped to one branch.
- **optimize** (no args) — full sweep across all namespaces. Use only at a natural session boundary or when explicitly asked. Do not call speculatively.
- **vacuum** — physical purge of stale/superseded chunks past retention. Always dry-run first (default). Never pass `dry_run: false` unless the user explicitly asks to delete.

Call `optimize` with a path after writing many chunks to a namespace. Global `optimize` only at end of session or on explicit request.

## How to read memory

When consuming context, read layers in reverse order — most distilled first. **Stop as soon as you have enough to answer.**

1. **Gold first** — authoritative conclusions, stable decisions, orientation. If Gold answers the question, stop here.
2. **Silver second** — working summaries and refined facts. Read only if Gold was too brief or missing. If Silver answers the question, stop here.
3. **Bronze last** — raw source material. Read only when you need the original wording, a timestamp, or a detail that Silver didn't preserve.

`session_start` already surfaces Gold. Call `read_context` only when Gold is insufficient for the task at hand.

## Mandatory write layering

Every new memory item must be written in this order:

1. **Bronze first** — write the raw fact, decision, observation, transcript, or session note exactly enough to preserve source context.
2. **Silver second** — only after Bronze exists, write a refined working summary derived from that Bronze record.
3. **Gold last** — only after Bronze and Silver exist, write a short durable insight with `append_gold_aspect` when the conclusion is stable enough to orient future sessions.

Never write directly to Silver when there is no Bronze source for the same fact or decision.
Never write directly to Gold as the first record of a new idea.
If using `batch_append`, order chunks Bronze before Silver in the batch, then call `append_gold_aspect` separately after the batch succeeds.

## User's namespace conventions

- `PROJECTS/*` — projects and technical details
- `META/*` — information about Vertical Brain itself
- Create new namespaces by topic: `WORK/ProjectName`, `LEARNING/Topic`, `DECISIONS/Area`

## Rules

1. Do not ask permission to write to memory — write proactively when you learn something important
2. Use `search` to find a specific fact
3. Use `read_context` to read everything stored under a specific namespace
4. The default write layer for new information is `bronze`; Silver and Gold are promotion layers, not first-write targets
5. Never run `vacuum` with `dry_run: false` unless the user explicitly requests deletion
6. Never run `rename_namespace` unless the user explicitly requests it — it is irreversible without manual intervention
7. Never call global `optimize` (no path) speculatively — only at session end or on explicit request
