# Vertical Brain Architecture Manifesto

## Definition

Vertical Brain is a personal knowledge management system where AI acts as a data architect, not just a chat assistant.

The system routes, stores, links, compacts, and retrieves knowledge using strict vertical namespace isolation.

## Storage model

Every knowledge item belongs to a path:

```text
ROOT / DOMAIN / PROJECT / OBJECT / TOPIC
```

Examples:

```text
WORK/DataArt/Databricks/Certification/StructuredStreaming
WORK/DataArt/FXDB/Payments/POP_Codes
WORK/PepsiCo/Experience
PERSONAL/Documents/RomanianCitizenship
TRADING/Strategies/HedgingModel
```

## Medallion layers

### Bronze

Raw data — append-only evidence log:

- chat fragments
- rough notes
- screenshots OCR
- copied code
- temporary observations

### Silver

Living current summary — one per namespace, kept up-to-date:

- normalized facts
- clarified rules
- corrected explanations
- grouped notes

Created once with `append_chunk(layer=silver)`. All subsequent updates use `update_silver` (OCC-protected). The optimizer may also write Silver via compaction.

### Gold

Stable search index — short routing anchors that help find the right namespace:

- semantic tags
- durable query labels
- architecture-rule anchors
- decision lookup labels
- reusable prompt labels

Requires an active Silver to exist at the same namespace. Written only via `append_gold_aspect`.
Silver holds the content; Gold holds many short, semantically distinct aspects that are embedded individually for routing.

## Read path

The read path must avoid global context pollution.

```text
Question
→ inspect namespace map
→ lock target vertical
→ retrieve bounded target context + ancestor summaries
→ expose horizontal links as handles
→ expand a link only when explicitly requested
→ answer
```

## Write path

The write path must route and compact new information. Bronze → Silver → Gold order is enforced by the infrastructure.

```text
Input
→ classify content
→ choose deepest target path
→ sibling scan (search before write)
→ conflict check
→ append_chunk (Bronze)
→ append_chunk(layer=silver) if first Silver, else update_silver (OCC)
→ append_gold_aspect only after active Silver exists
→ index
```

Invariants enforced at the executor level:
- Exact-duplicate Bronze is hard-blocked (content_hash check).
- Similar Bronze triggers a soft warning (similar_bronze in OperationResult).
- append_chunk(layer=silver) is blocked if an active Silver already exists.
- append_gold_aspect is blocked if no active Silver exists.
- Long Gold aspects are written but return a soft warning (aspect_too_long in OperationResult).

## Optimize process

OPTIMIZE is semantic compaction.

It should:

- merge small related chunks
- mark stale chunks
- update parent summaries
- preserve lineage to original raw chunks
- avoid merging unrelated contexts based only on generic keywords
