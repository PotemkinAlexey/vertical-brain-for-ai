from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4


Layer = Literal["bronze", "silver", "gold"]
ContentType = Literal["fact", "correction", "decision", "question", "note", "code", "artifact"]
ChunkStatus = Literal["active", "stale", "legacy", "superseded", "contradicted", "uncertain"]
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
class RouteDecision:
    target_path: str
    content_type: ContentType
    layer: Layer
    action: Action
    peer_links: list[PeerLinkCandidate] = field(default_factory=list)
    stale_candidates: list[StaleCandidate] = field(default_factory=list)
    confidence: float = 1.0
    reasoning_summary: str = ""
