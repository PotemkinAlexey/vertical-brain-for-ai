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

## What to write — and what not to

**Write** when you learn something that should survive this session: a decision, a confirmed fact, a project state change, a principle that emerged from discussion.

**Do not write**:
- Transient conversational exchanges with no lasting value
- Facts already present in memory (use `search` first — avoid duplicates)
- Uncertain guesses — if unsure, mark confidence low or wait for confirmation
- One giant chunk covering multiple unrelated facts — **one fact per chunk**

## Correcting wrong memory

Never delete or overwrite Gold directly. Instead:
1. Write a Bronze chunk with the correction and `content_type: "correction"`
2. Write a Silver summary that supersedes the old view
3. Only promote to Gold once the correction is confirmed stable

The old Gold aspect will be naturally displaced by the optimizer over time.

## At the end of every session

Call `session_end` with both fields — the optimizer cannot write Silver for you, it only does mechanical text compaction:
- `notes` — raw Bronze capture: bullet list of what was done, decided, or learned this session
- `summary` — refined Silver summary: one concise paragraph distilling the key outcome (you must write this, not the optimizer)
- `gold_aspect` — optional: only if a durable insight emerged that should orient future sessions

Then call `optimize` with the most-written namespace path.

## Mandatory write layering

Bronze is an append-only log. Silver is a living document you keep current. Gold is stable and rarely changes.

**The pattern for every new fact:**

1. **Bronze** — append the raw fact as-is. Never edit Bronze.
2. **Silver** — read the current Silver for this namespace, then write a new Silver that incorporates the new Bronze fact into the existing summary. Silver is always a complete up-to-date distillation, not a list of additions.
3. **Gold** — only when a conclusion is stable enough to orient future sessions, add it via `append_gold_aspect`.

This keeps cost low: each Silver update only needs the current Silver + one new Bronze chunk, not the full Bronze history.

Never write directly to Silver without a Bronze source for the same fact.
Never write directly to Gold as the first record of a new idea.
Never rely on the optimizer to produce Silver — it only does mechanical text compaction.
If using `batch_append`, order chunks Bronze before Silver in the batch, then call `append_gold_aspect` separately after the batch succeeds.

## User's namespace conventions

- `PROJECTS/*` — projects and technical details
- `META/*` — information about Vertical Brain itself
- Create new namespaces by topic: `WORK/ProjectName`, `LEARNING/Topic`, `DECISIONS/Area`

## Rules

1. Do not ask permission to write to memory — write proactively when you learn something important
2. **Search before you write** — use `search` to check if a fact already exists before creating a new chunk
3. Use `read_context` to read everything stored under a specific namespace
4. The default write layer for new information is `bronze`; Silver and Gold are promotion layers, not first-write targets
5. One fact per chunk — do not bundle multiple unrelated facts into one chunk
6. Never run `vacuum` with `dry_run: false` unless the user explicitly requests deletion
7. Never run `rename_namespace` unless the user explicitly requests it — it is irreversible without manual intervention
8. Never call global `optimize` (no path) speculatively — only at session end or on explicit request
