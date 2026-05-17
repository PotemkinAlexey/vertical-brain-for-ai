# Vertical Brain — persistent memory across sessions

## Principles

- **Not permitted — do not do it.**
- **Not certain — do not state it as fact.**
- **Writing — cite your basis.**
- **Changing — leave a trace.**
- **Deleting — only on explicit order.**

You have an MCP server called `vertical-brain` connected. It is the user's personal knowledge store — facts, decisions, and insights organized in hierarchical namespaces across Bronze/Silver/Gold data layers.

## At the start of every session

**Call `session_start` before responding to the user.** Do not begin the conversation until you have read the memory map. This is not optional — skipping it means you are operating blind.

It returns a map of all namespaces with active chunk counts and Gold insights. Read the response and briefly tell the user what you see in memory.

## How to read session_start output

- Namespaces in `CATEGORY/topic/subtopic` format
- Active chunk count next to each namespace — how many facts are stored
- Gold insights — distilled key conclusions

## When to write to memory

- **append_chunk** — record a raw Bronze fact, decision, or observation
- **update_silver** — atomically rewrite the Silver summary for a namespace (requires `current_silver_id` from your last read — OCC protected)
- **append_gold_aspect** — add a durable insight to Gold
- **batch_append** — write multiple Bronze chunks at once
- **session_end** — end-of-session: Bronze notes + Silver summary + optional Gold aspect

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

## Bronze dedup enforcement

The infrastructure enforces two levels of dedup on every `append_chunk` with `layer: bronze`:

1. **Hard block — exact duplicate.** If an active Bronze chunk with byte-for-byte identical content already exists at the same namespace, the write is rejected with a `ValueError`. The error message names the existing chunk ID and tells you to call `mark_stale` first if the fact has changed.

2. **Soft warning — similar content.** If the write succeeds but similar (not identical) active Bronze chunks exist at the same namespace, the response includes a `similar_bronze` field listing up to 3 matching chunks with snippets. Review them — if one is effectively the same fact, mark it stale to keep Silver clean.

```
result.similar_bronze = [
  { "chunk_id": "abc...", "snippet": "Databricks uses Delta Lake...", "score": 10 }
]
```

**Why this matters:** duplicate Bronze chunks make Silver dirty unnecessarily. If you keep Bronze unique, each `update_silver` synthesizes only genuinely new information.

## Correcting wrong memory

Never delete or overwrite Gold directly. Instead:
1. Write a Bronze chunk with the correction and `content_type: "correction"`
2. Write a Silver summary that supersedes the old view
3. Only promote to Gold once the correction is confirmed stable

The old Gold aspect will be naturally displaced by the optimizer over time.

## At the end of every session

**Call `session_end` before closing.** This is mandatory — not a suggestion. If the session produced any facts, decisions, or insights, they must be persisted. An unsaved session is a lost session.

`session_end` takes:
- `notes` — raw Bronze capture: bullet list of what was done, decided, or learned this session
- `summary` — refined Silver summary: one concise paragraph distilling the key outcome (you must write this — the optimizer cannot)
- `gold_aspect` — optional: only if a durable insight emerged that should orient future sessions

Then call `optimize` with the most-written namespace path.

## Mandatory write layering

Bronze is an append-only log. Silver is a living document you keep current. Gold is stable and rarely changes.

**The pattern for every new fact:**

1. **Bronze** — `append_chunk` the raw fact as-is. Never edit Bronze.
2. **Silver** — call `read_context` to get the current Silver chunk and its ID, then synthesize new Silver (current Silver + new Bronze fact), then call `update_silver(path, new_content, current_silver_id)`. Silver is always one complete up-to-date distillation per namespace.
3. **Gold** — only when a conclusion is stable enough to orient future sessions, call `append_gold_aspect`.

`update_silver` enforces OCC: it rejects the write if Silver changed since you read it, forcing a re-read and re-synthesis. This makes it impossible to overwrite someone else's update accidentally.

This keeps cost low: each Silver update only needs the current Silver + one new Bronze chunk, not the full Bronze history.

Never write directly to Gold as the first record of a new idea.
Never rely on the optimizer to produce Silver — it only does mechanical text compaction.
If writing multiple Bronze chunks at once, use `batch_append`, then call `update_silver` once with all new facts incorporated.

## User's namespace conventions

- `PROJECTS/*` — projects and technical details
- `META/*` — information about Vertical Brain itself
- Create new namespaces by topic: `WORK/ProjectName`, `LEARNING/Topic`, `DECISIONS/Area`

## Rules

### Write discipline

1. **Call `session_start` first.** Do not respond to the user until it returns. No exceptions.
2. **Call `session_end` last.** Every session that produced knowledge must end with `session_end`. Do not skip it even if the session felt lightweight.
3. **Search before every Bronze write.** Use `search` to check for existing chunks before calling `append_chunk`. If a matching chunk is found — stop. Do not write a duplicate.
4. **One fact per chunk.** Do not bundle multiple unrelated facts into one chunk. Each chunk must be independently meaningful and stale-able.
5. **After every Bronze write, update Silver.** Call `read_context` to get the current Silver and its ID, synthesize new Silver (old Silver + new fact), then call `update_silver`. Never leave Bronze orphaned without a Silver update.
6. **Call `read_context` before every `update_silver`.** You must pass `current_silver_id` — the ID of the Silver chunk you are replacing. Never call `update_silver` without having read the current Silver first.
7. **Never write Gold before Silver.** The infrastructure will reject it. If `append_gold_aspect` fails with "no active Silver found", call `update_silver` first, then retry.
8. **Never write Gold as the first record of a new idea.** Bronze → Silver → Gold, always in that order.

### Response to dedup signals

9. **If `append_chunk` is rejected with "identical content already exists"** — do not retry with the same content. Either the fact is already recorded (do nothing) or it changed (call `mark_stale` on the old chunk first, then write the corrected version).
10. **If `append_chunk` returns `similar_bronze`** — you must act. Call `mark_stale` on any listed chunk that is superseded by the new fact before proceeding to `update_silver`. Do not silently ignore the warning.

### Destructive operations

11. **Never run `vacuum` with `dry_run: false`** unless the user explicitly asks to delete data.
12. **Never run `rename_namespace`** unless the user explicitly requests it — it is irreversible without manual intervention.
13. **Never call global `optimize` (no path) speculatively** — only at session end or on explicit request.
