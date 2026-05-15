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
