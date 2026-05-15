from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


Layer = Literal["bronze", "silver", "gold"]
ContentType = Literal["fact", "correction", "decision", "question", "note", "code", "artifact"]
ChunkStatus = Literal["active", "stale", "legacy", "superseded", "contradicted", "uncertain"]
QueryType = Literal["explanation", "lookup", "comparison", "summary", "unknown"]
LinkExpansionPolicy = Literal["handles_only", "expanded", "none"]
OperationType = Literal[
    "create_node",
    "append_chunk",
    "create_link",
    "mark_stale",
    "supersede_chunk",
    "update_gold_summary",
]
Action = Literal[
    "append_bronze",
    "append_silver",
    "create_node",
    "append_and_optimize",
    "mark_stale",
    "update_gold",
    "ask_clarification",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonSerializable:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


@dataclass
class Node:
    path: str
    name: str
    id: str = field(default_factory=lambda: str(uuid4()))
    parent_path: str | None = None
    node_type: str = "default"
    gold_summary: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass
class Chunk:
    node_path: str
    content: str
    layer: Layer = "bronze"
    content_type: ContentType = "note"
    status: ChunkStatus = "active"
    source: str = "manual"
    confidence: float = 1.0
    lineage: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass
class Link:
    source_path: str
    target_path: str
    link_type: str
    reason: str
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)


@dataclass
class ChunkInput:
    content: str
    layer: Layer = "bronze"
    content_type: ContentType = "note"
    source: str = "model"
    confidence: float = 1.0
    lineage: list[str] = field(default_factory=list)


@dataclass
class LinkInput:
    target_path: str
    link_type: str
    reason: str


@dataclass
class StaleCandidateInput:
    path: str
    reason: str


@dataclass
class StorageOperation(JsonSerializable):
    operation: OperationType
    target_path: str
    chunk: ChunkInput | None = None
    links: list[LinkInput] = field(default_factory=list)
    stale_candidates: list[StaleCandidateInput] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    gold_summary: str | None = None
    confidence: float = 1.0
    reasoning_summary: str = ""


@dataclass
class StorageOperationBatch(JsonSerializable):
    operations: list[StorageOperation] = field(default_factory=list)
    reasoning_summary: str = ""


@dataclass
class OperationResult(JsonSerializable):
    operation: OperationType
    target_path: str
    chunk_id: str | None = None
    link_ids: list[str] = field(default_factory=list)
    stale_candidates: list[StaleCandidateInput] = field(default_factory=list)
    status: str = "applied"


@dataclass
class OperationBatchResult(JsonSerializable):
    results: list[OperationResult] = field(default_factory=list)
    status: str = "applied"


@dataclass
class ContextBudget:
    max_items: int = 12


@dataclass
class ContextPolicy:
    include_ancestors: bool = True
    include_target: bool = True
    link_expansion: LinkExpansionPolicy = "handles_only"


@dataclass
class ContextItem:
    path: str
    layer: str
    content: str
    source: str = "chunk"


@dataclass
class LinkHandle:
    link_id: str
    target_path: str
    link_type: str
    reason: str


@dataclass
class LockedContext(JsonSerializable):
    target_path: str
    items: list[ContextItem] = field(default_factory=list)
    link_handles: list[LinkHandle] = field(default_factory=list)
    budget: ContextBudget = field(default_factory=ContextBudget)
    policy: ContextPolicy = field(default_factory=ContextPolicy)
    omitted_items: int = 0

    def as_prompt_lines(self) -> list[str]:
        return [f"[{item.path}][{item.layer}] {item.content}" for item in self.items]


@dataclass
class PeerLinkCandidate:
    path: str
    reason: str


@dataclass
class StaleCandidate:
    path: str
    reason: str


@dataclass
class RouteDecision(JsonSerializable):
    target_path: str
    content_type: ContentType
    layer: Layer
    action: Action
    peer_links: list[PeerLinkCandidate] = field(default_factory=list)
    stale_candidates: list[StaleCandidate] = field(default_factory=list)
    confidence: float = 1.0
    reasoning_summary: str = ""


@dataclass
class AllowedContext:
    include_ancestors: bool = True
    include_peer_links: bool = True
    exclude_other_branches: bool = True


@dataclass
class QueryRouteDecision(JsonSerializable):
    target_path: str
    allowed_context: AllowedContext = field(default_factory=AllowedContext)
    query_type: QueryType = "lookup"
    confidence: float = 1.0
    reasoning_summary: str = ""
