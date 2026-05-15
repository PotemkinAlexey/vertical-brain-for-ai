# Domain Model

## Node

Represents a namespace path in the knowledge tree.

Fields:

```text
id
path
parent_id
name
node_type
gold_summary
created_at
updated_at
```

## Chunk

Represents an atomic knowledge item.

Fields:

```text
id
node_id
layer: bronze | silver | gold
content
content_type: fact | correction | decision | question | note | code | artifact
status: active | stale | legacy | superseded | contradicted | uncertain
source
confidence
created_at
updated_at
```

## Link

Represents explicit relationship between nodes or chunks.

Fields:

```text
id
source_id
target_id
link_type: peer | reference | dependency | contradiction | supersedes
reason
created_at
```

## RouteDecision

Compatibility adapter output from a model routing step. A RouteDecision should be
converted into a StorageOperation before mutating storage.

Fields:

```text
target_path
content_type
layer
action
peer_links
stale_candidates
confidence
reasoning_summary
```

## StorageOperation

Primary write contract returned by a model or produced by an adapter.

Fields:

```text
operation: create_node | append_chunk | create_link | mark_stale | supersede_chunk | update_gold_summary
target_path
chunk
links
stale_candidates
chunk_ids
gold_summary
confidence
reasoning_summary
```

`append_chunk` writes an atomic chunk to a vertical namespace and may create
explicit horizontal links. Stale candidates are surfaced but are not applied
automatically.

## StorageOperationBatch

Primary optimization contract for many related writes.

Fields:

```text
operations
reasoning_summary
```

Optimizers use batches to collapse many active variants into one canonical
Silver chunk, then supersede the original chunks while preserving lineage.

## Allowed actions

```text
append_bronze
append_silver
create_node
append_and_optimize
mark_stale
update_gold
ask_clarification
```
