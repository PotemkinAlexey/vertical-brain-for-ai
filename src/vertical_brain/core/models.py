from __future__ import annotations

import hashlib
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
    "append_gold_aspect",
    "rename_namespace",
    "update_silver",
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


STAGING_PATH = "STAGING/Unclassified"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OptimisticLockException(Exception):
    """Raised when a branch was mutated between plan_branch and apply_batch."""


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
    is_dirty: bool = False
    version: int = 0
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
    chunk_key: str | None = None
    content_hash: str = field(default="")
    supersedes: list[str] = field(default_factory=list)
    valid_from: str = field(default_factory=utc_now)
    valid_to: str | None = None
    decay_factor: float = 1.0

    def __post_init__(self) -> None:
        if not self.content_hash:
            normalized = " ".join(self.content.strip().split())
            self.content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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
    gold_aspect: str | None = None
    new_path: str | None = None
    confidence: float = 1.0
    reasoning_summary: str = ""
    current_silver_id: str | None = None
    source_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class StorageOperationBatch(JsonSerializable):
    operations: list[StorageOperation] = field(default_factory=list)
    reasoning_summary: str = ""
    branch_path: str | None = None
    start_version: int | None = None


@dataclass
class ValidationIssue(JsonSerializable):
    path: str
    message: str
    severity: str = "error"


@dataclass
class ValidationResult(JsonSerializable):
    valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)


@dataclass
class OperationResult(JsonSerializable):
    operation: OperationType
    target_path: str
    chunk_id: str | None = None
    link_ids: list[str] = field(default_factory=list)
    stale_candidates: list[StaleCandidateInput] = field(default_factory=list)
    status: str = "applied"
    validation: ValidationResult | None = None
    overflow_path: str | None = None
    # Populated when a Bronze chunk was written but similar active Bronze chunks already exist.
    # Not an error — agent should review and consider mark_stale on the similar chunks.
    similar_bronze: list["SearchResult"] = field(default_factory=list)
    # True when a Bronze chunk was written but its content exceeds the recommended size for
    # quality embeddings. Not an error — but the agent should split the content into
    # smaller single-fact chunks so each one embeds meaningfully.
    chunk_too_large: bool = False


@dataclass
class OperationBatchResult(JsonSerializable):
    results: list[OperationResult] = field(default_factory=list)
    status: str = "applied"
    validation: ValidationResult | None = None


@dataclass
class SearchResult(JsonSerializable):
    path: str
    source: str
    score: float
    snippet: str
    chunk_id: str | None = None
    layer: str | None = None
    content_type: str | None = None
    status: str | None = None


@dataclass
class SearchCandidateHandle(JsonSerializable):
    path: str
    source: str
    score: float
    chunk_id: str | None = None
    layer: str | None = None
    content_type: str | None = None
    status: str | None = None


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
class LinkExpansionResult(JsonSerializable):
    link_id: str
    source_path: str
    expanded_path: str
    locked_context: LockedContext
    link_type: str = ""
    reason: str = ""


@dataclass
class NamespaceMapNode(JsonSerializable):
    path: str
    name: str
    parent_path: str | None
    depth: int
    children: list[str] = field(default_factory=list)
    chunk_count: int = 0
    active_chunk_count: int = 0
    stale_chunk_count: int = 0
    subtree_chunk_count: int = 0
    link_count: int = 0
    link_handles: list[LinkHandle] = field(default_factory=list)
    gold_summary: str = ""
    omitted_summary_chars: int = 0
    updated_at: str = ""


@dataclass
class NamespaceMap(JsonSerializable):
    root_path: str | None = None
    nodes: list[NamespaceMapNode] = field(default_factory=list)
    omitted_nodes: int = 0
    summary_max_chars: int = 240


@dataclass
class SearchContextResult(JsonSerializable):
    query: str
    candidate_handles: list[SearchCandidateHandle] = field(default_factory=list)
    locked_contexts: list[LockedContext] = field(default_factory=list)
    omitted_candidates: int = 0


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


@dataclass
class EmbeddingRouteCandidate:
    path: str
    score: float
    gold_summary: str
