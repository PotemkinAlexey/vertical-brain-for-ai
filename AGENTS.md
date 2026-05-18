# Vertical Brain — agent contract

## Principles

- **Not permitted — do not do it.**
- **Not certain — do not state it as fact.**
- **Writing — cite your basis.**
- **Changing — leave a trace.**
- **Deleting — only on explicit order.**

## Session

**Call `session_start` before responding. No exceptions.** If it fails, tell the user and stop.

**Call `session_end` if the session produced durable knowledge.** If nothing worth persisting happened, tell the user explicitly: "No durable memory was created this session."

## Reading memory

Gold = routing only (short search tags, not answers). Use Gold to find which namespace to read, then call `read_context` to get Silver. Silver is the answer. Bronze only when you need the original wording or a timestamp Silver didn't preserve.

`session_start` surfaces Gold. Always follow with `read_context` on relevant namespaces before answering questions about stored knowledge.

## Writing during a session

**Write as you go — do not save everything for session_end.** When a decision is made, a feature is built, or a fact is confirmed mid-session: write it immediately (Bronze + Silver update). `session_end` is a final summary, not the only write point.

## Writing memory

1. **Search before every Bronze write.** If a matching chunk exists — stop, do not duplicate.
2. **One fact per chunk.** Each chunk must be independently meaningful and stale-able.
3. **Bronze → Silver → Gold, always in this order.** After every Bronze write, update Silver. Never write Gold as the first record of a new idea.
4. **Call `read_context` before every `update_silver`** to get `current_silver_id`.
5. **For `batch_append`:** call `update_silver` once after the entire batch, not per chunk.
6. **To correct wrong memory:** write a Bronze chunk with `content_type="correction"`, then update Silver. Do not overwrite Gold directly.
7. **Canonical namespace first.** Durable knowledge belongs in its canonical topic namespace. Facts about a current storage, project, source, workstream, or other concrete topic must be written inside that topic's context. `META/agent-contract` is only for general operating rules; `META/sessions/*` is only a trace. `session_end` does not replace Bronze → Silver → Gold updates in the relevant namespace. If placement is unclear, route/read context before writing.

## Responding to tool signals

- `append_chunk` rejected (identical content) — do not retry. Fact already recorded, or mark_stale the old one first.
- `similar_bronze` returned — review. Mark stale only if the older chunk is superseded.
- `chunk_too_large: true` — split into single-fact chunks, then update Silver once.
- `aspect_too_long: true` — split into shorter search tags (target 30–100 chars each).

## Destructive operations

- **Never `vacuum(dry_run=false)`** unless the user explicitly asked.
- **Never `rename_namespace`** unless explicitly requested.
- **Never call global `optimize()` speculatively** — only at session end or on explicit request.
- **Never mutate storage outside the MCP protocol** (no raw SQL, no direct file edits).
- If the user says "clean", "wipe", "reset", "delete" — confirm intent before acting.

## File ingestion

When user writes `ingest_file` or `ingest_url`, call the tool and follow the **IRON RULES** protocol it returns inline. Do not respond to the user until `complete_ingest` succeeds.

- PDF: `ingest_file(source_path=<path>, authority=<inferred>)` — default `mode=answer_complete`
- Other formats: agent reads file → `ingest_file(content=<text>, file_name=<name>, authority=<inferred>)`
- URL: `ingest_url(url=<url>)` — same stateful session as `ingest_file`

**Success criterion:** the source file will not exist later. Any question the document should answer must be answerable from brain alone.

**Mandatory steps (server-enforced in `answer_complete` mode):**
1. Artifact Bronze (immutable registration) at `SOURCES/{slug}`
2. `[INVENTORY]` Bronze listing every answer-critical entity
3. Plan sub-namespace structure — scan `section_header` chunks → map sections to `SOURCES/{slug}/section-slug/` sub-namespaces; write plan as Bronze note at root
4. Process **every** service chunk in batches by section (`get_service_chunks` + `batch_mark_service_chunks`); write Bronze **verbatim** into the section sub-namespace
5. `finish_bronze_extraction` — max 25% skipped, min 50% extracted, zero extracted forbidden
6. Silver per sub-namespace = `[chunk_id] one-line summary` index (pointer map, not synthesis); root Silver = `[SOURCES/{slug}/section] description` section index
7. `submit_inventory_probes` — spot-check ≥30% of inventory items (min 3) with Bronze `chunk_id` citations
8. `complete_ingest` — verifies artifact + inventory + facts + Silver + probes in storage

`force=true` auto-wipes the target `SOURCES/{slug}` namespace (including immutable chunks) before re-ingest. Sessions persist across MCP restarts until `complete_ingest`.

Use `mode=routing` only when the user explicitly wants discoverability-only ingest (lighter rules).

`skipped` is only for true boilerplate (headers, footers, blank pages, disclaimers). If you cannot finish, report blockers — do not call `complete_ingest`.

## Schema / normative lookups

Answer field and schema questions only from an immutable Bronze chunk (cited via Silver `chunk_id`). If no immutable chunk matches — say so. Do not answer from model training data.

## Namespace conventions

- `PROJECTS/*` — projects and technical details
- `META/*` — Vertical Brain itself
- `SOURCES/{slug}` — ingested source documents; authority belongs in source metadata, not the path
- New topics: `WORK/Name`, `LEARNING/Topic`, `DECISIONS/Area`
