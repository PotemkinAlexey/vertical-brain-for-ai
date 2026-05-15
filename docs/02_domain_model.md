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

Structured output from the router.

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
