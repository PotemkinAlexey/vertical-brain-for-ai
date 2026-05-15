# Router JSON Contract

The router must always return strict JSON.

## Ingest contract

```json
{
  "target_path": "WORK/DataArt/Databricks/Certification/StructuredStreaming/SchemaEvolution",
  "content_type": "correction",
  "layer": "silver",
  "action": "append_and_optimize",
  "peer_links": [
    {
      "path": "WORK/DataArt/Databricks/Certification/AutoLoader/SchemaEvolution",
      "reason": "Source-side Auto Loader schema evolution is often confused with Delta sink schema evolution"
    }
  ],
  "stale_candidates": [
    {
      "path": "WORK/DataArt/Databricks/Certification/StructuredStreaming/SchemaEvolution",
      "reason": "Older note may incorrectly suggest cloudFiles.schemaEvolutionMode applies to Delta sink writes"
    }
  ],
  "confidence": 0.91,
  "reasoning_summary": "The input corrects schema evolution behavior for streaming Delta sink writes."
}
```

## Ask contract

```json
{
  "target_path": "WORK/DataArt/Databricks/Certification/StructuredStreaming",
  "allowed_context": {
    "include_ancestors": true,
    "include_peer_links": true,
    "exclude_other_branches": true
  },
  "query_type": "explanation",
  "confidence": 0.88,
  "reasoning_summary": "The question is about Databricks Structured Streaming exam knowledge."
}
```

## Rules

- Never route based only on generic keywords.
- Prefer the deepest relevant node.
- If two paths are possible, return candidates.
- If confidence is below 0.65, ask for clarification.
- Peer-links must be justified.
- Stale candidates must not be deleted automatically.
