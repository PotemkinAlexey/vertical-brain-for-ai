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

### Read-path with recall assist (v1.5–v1.8)

The full flow is **Gold routes → Silver answers → Bronze cites → Vector assists**:

1. `route(text=...)` → namespace candidates from Gold. Each candidate carries `match_source`:
   - `"gold"` — hit came from a Gold aspect (or path-token overlap fallback).
   - `"content_fallback"` — v1.7 rescue: Gold missed, but a Bronze/Silver semantic sweep found the namespace. Tunable via `fallback_threshold` (default 0.55; set negative to disable). Read the `next_hint`.
2. `read_context(path=<top>, query=<user question>)` → Silver answer + ancestor Gold. The response carries `silver_confidence` (`missing` | `low` | `weak` | `ok`):
   - `weak` means **both** lexical token overlap **and** (when an embedding endpoint is configured) cosine `>= 0.5` agreed the Silver is off-topic — a safe signal to widen, no false positives on synonyms.
   - `ok` means the Silver answers — do not run extra searches.
   - Items dropped by the budget come back as `omitted_chunk_ids`; pull them with `list_chunks` instead of re-issuing `read_context` with a higher `max_items`.
3. When Silver is `missing/low/weak`, call `search_semantic(query=..., root_path=<top>)`. The response is `{results, suggested_paths, semantic_endpoint}`. Results are biased toward Bronze evidence (reference > fact > decision); read the suggested namespace and cite the Bronze `chunk_id` in your answer.
4. Vector search never replaces Gold/Silver — it only helps you find the right Bronze across the namespace tree.

Both `route` and `search_semantic` report `semantic_endpoint: true|false`. When `false`, no embedding endpoint is configured and recall is bag-of-words overlap; the same calls become real semantic matching automatically once an endpoint (e.g. Ollama) is set, with no client changes needed.

## Writing during a session

**Write as you go — do not save everything for session_end.** When a decision is made, a feature is built, or a fact is confirmed mid-session: write it immediately (Bronze + Silver update). `session_end` is a final summary, not the only write point.

## Writing memory

1. **Search before every Bronze write.** If a matching chunk exists — stop, do not duplicate.
2. **One fact per chunk.** Each chunk must be independently meaningful and stale-able.
3. **Bronze → Silver → Gold, always in this order.** After every Bronze write, update Silver. Never write Gold as the first record of a new idea.
4. **Call `read_context` before every `update_silver`** — the Silver item's `chunk_id` in the response is the `current_silver_id` to pass.
5. **For `batch_append`:** call `update_silver` once after the entire batch, not per chunk.
6. **To correct wrong memory:** write a Bronze chunk with `content_type="correction"`, then update Silver. Do not overwrite Gold directly.
7. **Canonical namespace first.** Durable knowledge belongs in its canonical topic namespace. Facts about a current storage, project, source, workstream, or other concrete topic must be written inside that topic's context. `META/agent-contract` is only for general operating rules; `META/sessions/*` is only a trace. `session_end` does not replace Bronze → Silver → Gold updates in the relevant namespace. If placement is unclear, route/read context before writing.
8. **Silver is a current summary, not a changelog; Gold is a signpost, not a log.** Keep Silver concise and Gold to a few stable routing tags. When a namespace outgrows them, decompose it into sub-namespaces — do not let one Silver or Gold grow without bound. History and detail live in Bronze.

## Responding to tool signals

- `append_chunk` rejected (identical content) — do not retry. Fact already recorded, or mark_stale the old one first.
- `similar_bronze` returned — review. Mark stale only if the older chunk is superseded.
- `chunk_too_large: true` — split into single-fact chunks, then update Silver once.
- `aspects_too_long` returned (list of tags) — split each into shorter search tags (target 30–100 chars each).
- `silver_too_large` / `gold_near_limit` — the namespace is overloaded. Decompose it: move detail into sub-namespaces, each with its own focused Silver, and leave a Silver index of the children here. Call `doctor` for cluster-based split suggestions (its `namespace_overloaded` finding).

## Destructive operations

- **Never `vacuum(dry_run=false)`** unless the user explicitly asked.
- **Never `rename_namespace`** unless explicitly requested.
- **Never call global `optimize()` speculatively** — only at session end or on explicit request.
- **Never mutate storage outside the MCP protocol** (no raw SQL, no direct file edits).
- If the user says "clean", "wipe", "reset", "delete" — confirm intent before acting.

## File ingestion

When user writes `ingest_file` or `ingest_url`, call the tool and follow the **IRON RULES** protocol it returns inline. Do not respond to the user until `complete_ingest` succeeds.

**Before calling `ingest_file` or `ingest_url`, ask the user:**
> "How thoroughly should I ingest this?
> - **quick** — fast scan, up to 50% skip allowed, 60% coverage required
> - **standard** — balanced (default), 25% max skip, 90% coverage
> - **thorough** — deep extraction, 10% max skip, 95% coverage, stricter probes"

If the user already said something like "quickly", "just route it", or "full extraction" — infer the depth without asking.

- PDF/DOCX/DOC/HTML/ODT: `ingest_file(source_path=<path>, authority=<inferred>, depth=<chosen>)`
- Other formats: agent reads file → `ingest_file(content=<text>, file_name=<name>, authority=<inferred>, depth=<chosen>)`
- URL: `ingest_url(url=<url>, depth=<chosen>)` — same stateful session as `ingest_file`

**Success criterion:** the source file will not exist later. Any question the document should answer must be answerable from brain alone.

**Mandatory steps (server-enforced in `answer_complete` mode):**
1. Artifact Bronze (immutable registration) at `SOURCES/{slug}`
2. `[INVENTORY]` Bronze — each line = **one answer-critical capability**, not a category.
   Format: `"<Topic> — <specific fact or method>"`. Examples: `"US ACH payment to BofA"`, `"Germany EUR IBAN wire"`, `"Costa Rica MT103 SWIFT intermediary"`. Never write `"United States"` or `"payment methods"` — too vague to probe.
3. Plan sub-namespace structure — scan `section_header` chunks → map sections to `SOURCES/{slug}/section-slug/` sub-namespaces; write plan as Bronze note at root
4. Process **every** service chunk in batches by section (`get_service_chunks` + `batch_mark_service_chunks`); write Bronze into the section sub-namespace:
   - `content_type="reference"`, `immutable=True` — normative tables, SWIFT/BIC/IBAN/routing/account numbers
   - `content_type="fact"`, `immutable=True` recommended — prose instructions, rules, constraints
5. `finish_bronze_extraction` — thresholds depend on `depth` (shown in session header)
6. Silver per sub-namespace = `[chunk_id] one-line summary` index (pointer map, not synthesis); root Silver = `[SOURCES/{slug}/section] description` section index
7. `submit_inventory_probes` — coverage requirement depends on `depth` (shown in session header). Primary quality gate: `complete_ingest` will fail if coverage is below threshold.
8. `complete_ingest` — verifies artifact + inventory + facts + Silver + probes + coverage score; returns `coverage_score.uncovered_items` if any gaps remain
9. `append_gold_aspect` — **MANDATORY** after complete_ingest: add routing tags on source namespace and all sub-namespaces. Without Gold the document is invisible to routing.

`force=true` auto-wipes the target `SOURCES/{slug}` namespace (including immutable chunks) before re-ingest. Sessions persist across MCP restarts until `complete_ingest`.

`mode=routing` — discoverability-only; complete_ingest writes a `[ROUTING ONLY]` Bronze note. Not for authoritative answers.
`mode=audit` — check coverage gaps on an already-ingested namespace: submit_inventory_probes → complete_ingest. No Bronze writing.

`skipped` is only for true boilerplate (headers, footers, blank pages, disclaimers). If you cannot finish, report blockers — do not call `complete_ingest`.

## Schema / normative lookups

Answer field and schema questions only from an immutable Bronze chunk (cited via Silver `chunk_id`). If no immutable chunk matches — say so. Do not answer from model training data.

## Namespace conventions

- `PROJECTS/*` — projects and technical details
- `META/*` — Vertical Brain itself
- `SOURCES/{slug}` — ingested source documents; authority belongs in source metadata, not the path
- New topics: `WORK/Name`, `LEARNING/Topic`, `DECISIONS/Area`
