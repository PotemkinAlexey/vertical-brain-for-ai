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

If `session_start` fails because the MCP server is unavailable, tell the user explicitly and wait for confirmation before continuing. Do not proceed without memory on your own judgment.

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
- **mark_stale** — mark chunks stale by ID, or all active non-gold chunks at a path. Parameters:
  - `recursive: true` — includes all descendant namespaces (e.g. whole subtree).
  - `include_immutable: true` — also marks immutable chunks stale. Use **only** when wiping a namespace for re-ingestion. Without this flag immutable chunks are silently skipped.
  - Full wipe before re-ingestion: `mark_stale(path=..., recursive=true, include_immutable=true, reason="re-ingestion")` then `vacuum(dry_run=false)`.

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

3. **Soft warning — chunk too large.** If the write succeeds but the Bronze content exceeds 600 characters, the response includes `chunk_too_large: true`. This means the chunk likely bundles multiple facts and its embedding will be diluted across unrelated topics, degrading search and routing quality.

```
result.chunk_too_large = true
```

**What to do:** split the content into smaller single-fact chunks. Each chunk should express one idea, decision, or observation — independently meaningful and stale-able on its own.

## Correcting wrong memory

Never delete or overwrite Gold directly. Instead:
1. Write a Bronze chunk with the correction and `content_type: "correction"`
2. Write a Silver summary that supersedes the old view
3. Only promote to Gold once the correction is confirmed stable

The old Gold aspect will remain until explicitly replaced by a new `append_gold_aspect` call with updated content — the optimizer does not modify Gold.

## At the end of every session

**Call `session_end` before closing** if the session produced any facts, decisions, or insights. An unsaved session is a lost session.

If the session produced no durable knowledge (e.g. a quick lookup, a clarifying question, or a read-only task), you may skip `session_end` — but tell the user explicitly: "No durable memory was created this session."

`session_end` takes:
- `notes` — raw Bronze capture: bullet list of what was done, decided, or learned this session
- `summary` — refined Silver summary: one concise paragraph distilling the key outcome (you must write this — the optimizer cannot)
- `gold_aspect` — optional: only if a durable insight emerged that should orient future sessions

Then call `optimize` with the most-written namespace path.

## Mandatory write layering

Bronze is an append-only log. Silver is a living document you keep current. Gold is stable and rarely changes.

**The pattern for every new fact:**

1. **Bronze** — `append_chunk` the raw fact as-is. Never edit Bronze.
2. **Silver** — if no active Silver exists yet, create it with `append_chunk(layer="silver")`. For all subsequent updates: call `read_context` to get the current Silver and its ID, synthesize new Silver (current Silver + new Bronze fact), then call `update_silver(path, new_content, current_silver_id)`. Silver is always one complete up-to-date distillation per namespace.
3. **Gold** — only when a conclusion is stable enough to orient future sessions, call `append_gold_aspect`.

`update_silver` enforces OCC: it rejects the write if Silver changed since you read it, forcing a re-read and re-synthesis. This makes it impossible to overwrite someone else's update accidentally.

This keeps cost low: each Silver update only needs the current Silver + one new Bronze chunk, not the full Bronze history.

Never write directly to Gold as the first record of a new idea.
Never rely on the optimizer to produce Silver — it only does mechanical text compaction.
If writing multiple Bronze chunks at once, use `batch_append`, then call `update_silver` once with all new facts incorporated.

## Gold aspect guidance

Gold is a search index, not a knowledge summary. Silver holds the content; Gold holds short semantic anchors that help route future queries to the right namespace.

A Gold aspect should answer: **"Which query should find this namespace?"** Prefer many short, precise aspects over one long overview. Target ~30–100 characters per aspect. Avoid broad paragraphs like "what I know about X"; write tags like "Delta Lake Z-ordering lookup" or "AutoLoader schema drift handling".

`append_gold_aspect` returns a soft warning when an aspect is too long:

```
result.aspect_too_long = true
```

**What to do:** split the long aspect into smaller, semantically distinct search tags and append them separately.

## File ingestion protocol

### Triggers

**When the user writes `ingest_file` (with a file attached):**
1. Take the file content and filename from the attachment — do not ask the user for parameters.
2. Try to infer `authority` from the content (issuing organisation, standard body, domain name in the header, copyright line, etc.). If you cannot infer it with confidence, leave it empty — the namespace will be `SOURCES/{slug}`.
3. Call `ingest_file(content=<attachment text>, file_name=<attachment name>, authority=<inferred or empty>)`.
4. Then execute all steps below.

**When the user writes `ingest_url <url>`:**
1. Call `ingest_url(url=<url>)`. Authority and slug are derived automatically from the hostname and URL path.
2. Then execute all steps below.

The user never needs to pass parameters manually. One word is enough.

**Critical:** after calling `ingest_file` or `ingest_url`, execute all ingestion steps immediately and silently. Do **not** respond to the user with a summary, do **not** ask for confirmation, do **not** wait. Only report back once every step is complete.

---

Use this protocol whenever you ingest a file or URL. The tool returns a metadata header. You must execute all steps below — the tool does not do it for you.

### What immutable means

`immutable: true` marks a Bronze chunk as a permanent, verbatim record that must never change. Use it for:

- **Normative rules** — schema field definitions, mandatory constraints, code values defined by the issuing authority.
- **The source registration chunk** — the anchor chunk that records what document was ingested (name, hash, authority, version).
- **Verbatim extracts** — exact quoted text that must not be paraphrased (regulatory wording, contractual clauses, wire format specs).

Do **not** use `immutable: true` for:
- Your own interpretation or derived rules.
- Contextual notes and explanatory text you added.
- Anything you are not certain is verbatim from the source.

Immutable chunks are protected by default — `mark_stale` and `vacuum` skip them silently. To wipe a namespace completely for re-ingestion, use `mark_stale(include_immutable=true)`. In normal operation, if a newer version of the document changes a field, write a new Bronze chunk with `content_type: "correction"` and reference both the old and new immutable chunk IDs in the text.

### Referencing immutable chunks from Silver

Silver must never copy verbatim content from immutable Bronze. Instead, **reference the immutable chunk by ID**:

```
The MT103 field 32A carries value date, currency, and amount
(source: chunk_id=<id of the immutable Bronze chunk>).
```

When `append_chunk` or `batch_append` returns a chunk ID, record it immediately. Use that ID in the Silver narrative and in any derived Silver written to `WORK/` namespaces. This creates a traceable chain: WORK Silver → SOURCES Bronze → original document.

### Step-by-step

**Step 1 — Register the source document (always first)**

Write one `immutable: true` Bronze chunk at `SOURCES/{authority}/{slug}`:
- `layer: "bronze"`, `content_type: "artifact"`, `immutable: true`
- Content: file name, SHA-256 from the tool output, authority, document version or date if present, a one-sentence description. Max 600 characters.
- Record the returned chunk ID — every subsequent Silver update references this chunk as provenance.

**Step 2 — Extract atomic Bronze facts first**

For each distinct rule, definition, field, constraint, or code value in the document:
- One chunk = one fact. Max 600 characters. If a fact is longer, split it.
- Include source location: section heading, field tag, table name, paragraph number.
- `immutable: true` for schema fields, mandatory constraints, and normative rules copied verbatim.
- `immutable: false` for contextual notes and explanatory passages.
- `content_type: "fact"` for confirmed statements; `content_type: "question"` for ambiguous or contradictory passages.

Use `batch_append` for bulk extraction — **max 10 chunks per call**. For large documents, split into multiple `batch_append` calls (e.g. per section or per country). Collect every returned chunk ID.

**Step 3 — Write Silver from Bronze, never from the source**

⛔ Do NOT write Silver by summarising the source document directly. Silver must be synthesized exclusively from the Bronze chunks you just wrote.

After all Bronze batches are complete:
1. Call `list_chunks(path=..., layer="bronze")` to read back all the Bronze chunks you wrote.
2. Synthesize Silver from those chunks — not from your memory of the PDF/page.
3. Call `update_silver` (or `append_chunk(layer="silver")` if none exists) with `source_chunk_ids` set to the IDs of every Bronze chunk it covers.

This ensures the Silver→Bronze dependency graph is honest and verifiable. If you skip this step, Silver is ungrounded fiction.

**Step 4 — Write interpretation to WORK/ Silver, not Bronze**

If a source fact implies an implementation rule, usage guideline, or constraint for your project:
- Write it as Silver at the relevant `WORK/` or `PROJECTS/` namespace.
- Reference the source Bronze chunk ID(s) in the Silver text.
- Never write your interpretation as Bronze at SOURCES/ — Bronze there is only for what the document literally says.

**Step 5 — Handle conflicts explicitly**

If any extracted fact contradicts existing memory at a `WORK/` namespace:
- Do not overwrite or mark stale the existing chunk automatically.
- Write a new Bronze chunk with `content_type: "correction"` that names both the old fact (include its chunk ID if known) and the new authoritative version with source location.
- Report every conflict in your final summary to the user.

**Step 6 — Link namespaces**

After writing to both `SOURCES/...` and `WORK/...`:
```
create_link(source_path="WORK/...", target_path="SOURCES/{authority}/{slug}", link_type="derived_from")
```

**Step 7 — Gold only for stable orientation**

Add Gold aspects at `SOURCES/{authority}/{slug}` only if this document is a major anchor that should orient future routing for this namespace. Maximum two aspects. Write them as short search tags, not summaries.

### Namespace placement

| Content | Where |
|---|---|
| Source registration, verbatim schema fields, normative rules | `SOURCES/{authority}/{slug}` (immutable Bronze) |
| Contextual notes and explanatory text | `SOURCES/{authority}/{slug}` (mutable Bronze) |
| Silver overview of the document | `SOURCES/{authority}/{slug}` (Silver) |
| Derived implementation rules and guidelines | relevant `WORK/` or `PROJECTS/` namespace (Silver) |

### Report when done

Give the user a structured summary:
- **Source namespace** and registration chunk ID
- **Bronze facts** extracted: how many immutable vs mutable
- **Silver** at SOURCES/ and at WORK/ namespaces updated
- **Links** created
- **Conflicts** found (list each with old and new chunk IDs)
- **Open questions** (content_type=question chunks written)
- **Skipped content** and reason

---

## User's namespace conventions

- `PROJECTS/*` — projects and technical details
- `META/*` — information about Vertical Brain itself
- Create new namespaces by topic: `WORK/ProjectName`, `LEARNING/Topic`, `DECISIONS/Area`

## Rules

### Write discipline

1. **Call `session_start` first.** Do not respond to the user until it returns. No exceptions.
2. **Call `session_end` last** if the session produced durable knowledge. If nothing worth persisting happened, skip it — but tell the user explicitly that no memory was created.
3. **Search before every Bronze write.** Use `search` to check for existing chunks before calling `append_chunk`. If a matching chunk is found — stop. Do not write a duplicate.
4. **One fact per chunk.** Do not bundle multiple unrelated facts into one chunk. Each chunk must be independently meaningful and stale-able.
5. **After every Bronze write, update Silver.**
   - If no active Silver exists yet at this namespace, create the first one with `append_chunk(layer="silver")`.
   - If an active Silver already exists, call `read_context` to get its ID, synthesize new Silver (old Silver + new fact), then call `update_silver`. Never leave Bronze orphaned without a Silver update.
   - For `batch_append`: update Silver **once** after the entire batch succeeds, incorporating all new Bronze facts together. Do not call `update_silver` separately for each chunk.
6. **Call `read_context` before every `update_silver`.** You must pass `current_silver_id` — the ID of the Silver chunk you are replacing. Never call `update_silver` without having read the current Silver first.
7. **Never write Gold before Silver.** The infrastructure will reject it. If `append_gold_aspect` fails with "no active Silver found", call `update_silver` first, then retry.
8. **Never write Gold as the first record of a new idea.** Bronze → Silver → Gold, always in that order.

### Response to dedup signals

9. **If `append_chunk` is rejected with "identical content already exists"** — do not retry with the same content. Either the fact is already recorded (do nothing) or it changed (call `mark_stale` on the old chunk first, then write the corrected version).
10. **If `append_chunk` returns `similar_bronze`** — you must review and decide. Do not ignore the warning. Mark stale only if the older chunk is actually superseded or contradicted by the new fact. If both chunks are still valid (different aspects of the same topic), keep both and mention this in the Silver update.
11. **If `append_chunk` returns `chunk_too_large: true`** — split the content into smaller single-fact chunks. Write each fact as a separate `append_chunk` call, then call `update_silver` once with all of them incorporated. A chunk that embeds poorly is invisible to routing and search.
12. **If `append_gold_aspect` returns `aspect_too_long: true`** — split it into shorter search tags. Gold aspects should be routing anchors, not summaries.

### Destructive operations

13. **Never run `vacuum` with `dry_run: false`** unless the user explicitly asks to delete data.
14. **Never run `rename_namespace`** unless the user explicitly requests it — it is irreversible without manual intervention.
15. **Never call global `optimize` (no path) speculatively** — only at session end or on explicit request.
