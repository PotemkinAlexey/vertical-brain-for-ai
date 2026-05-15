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

Raw data:

- chat fragments
- rough notes
- screenshots OCR
- copied code
- temporary observations

### Silver

Cleaned structured knowledge:

- normalized facts
- clarified rules
- corrected explanations
- grouped notes

### Gold

Stable high-level knowledge:

- summaries
- decisions
- architecture rules
- durable facts
- exam cheat sheets
- reusable prompts

## Read path

The read path must avoid global context pollution.

```text
Question
→ classify target vertical
→ lock unrelated branches
→ retrieve target node + ancestors + approved peer-links
→ answer
```

## Write path

The write path must route and compact new information.

```text
Input
→ classify content
→ choose deepest target path
→ sibling scan
→ conflict check
→ write Bronze or Silver
→ update Gold summary if needed
→ index
```

## Optimize process

OPTIMIZE is semantic compaction.

It should:

- merge small related chunks
- mark stale chunks
- update parent summaries
- preserve lineage to original raw chunks
- avoid merging unrelated contexts based only on generic keywords
