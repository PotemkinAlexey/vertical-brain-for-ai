from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4


class Layer(StrEnum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class ChunkStatus(StrEnum):
    ACTIVE = "active"
    STALE = "stale"
    LEGACY = "legacy"
    SUPERSEDED = "superseded"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


class OperationType(StrEnum):
    CREATE_NODE = "create_node"
    APPEND_CHUNK = "append_chunk"
    CREATE_LINK = "create_link"
    MARK_STALE = "mark_stale"
    SUPERSEDE_CHUNK = "supersede_chunk"
    APPEND_GOLD_ASPECT = "append_gold_aspect"
    RENAME_NAMESPACE = "rename_namespace"
    UPDATE_SILVER = "update_silver"


ContentType = Literal["fact", "reference", "correction", "decision", "question", "note", "code", "artifact"]
QueryType = Literal["explanation", "lookup", "comparison", "summary", "unknown"]
LinkExpansionPolicy = Literal["handles_only", "expanded", "none"]
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
    # v1.12 extension slot. Free-form dict for enterprise data the open core
    # neither reads nor writes (tenant_id, geo_residency, billing_owner, etc.).
    # Defaults to `{}`; serialized as JSON by storage backends.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    node_path: str
    content: str
    layer: Layer = Layer.BRONZE
    content_type: ContentType = "note"
    status: ChunkStatus = ChunkStatus.ACTIVE
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
    immutable: bool = False
    # Step 3 usage telemetry: counters bumped by the read path so that
    # promotion decisions (Bronze→Silver→Gold) and Step 4 reputation
    # scoring can use observed usefulness instead of operator intuition.
    # `access_count` and `last_accessed` are updated on every read that
    # surfaces this chunk's id via `bump_chunk_access`. `last_positive_use`
    # is reserved for Step 4 — wired from the Birch resonance-feedback
    # path; Step 3 only ships the column.
    access_count: int = 0
    last_accessed: str | None = None
    last_positive_use: str | None = None
    # v1.12 extension slot for enterprise (classification, tenant_id, retention
    # policy id, geo zone, source-system record id, etc.). The open core does
    # not read or interpret these fields — they survive write/read round-trips
    # via the storage backends and are intended to be consulted by
    # `chunk_filter` callbacks (v1.11) and reranker / connector layers.
    metadata: dict[str, Any] = field(default_factory=dict)

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
    layer: Layer = Layer.BRONZE
    content_type: ContentType = "note"
    source: str = "model"
    confidence: float = 1.0
    lineage: list[str] = field(default_factory=list)
    immutable: bool = False
    # v1.12 — optional client-supplied metadata forwarded into the saved Chunk.
    metadata: dict[str, Any] = field(default_factory=dict)


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
    force_immutable: bool = False  # override immutable protection for mark_stale (wipe/re-ingestion)


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
    # True when a Gold aspect was written but is longer than the recommended size for
    # precise routing. Not an error — Gold aspects are search tags and should stay short.
    aspect_too_long: bool = False
    # True when the Silver written exceeds the recommended summary size. Not an error —
    # signals the namespace is overloaded and should be decomposed into sub-namespaces.
    silver_too_large: bool = False
    # True when a node's Gold aspect count is approaching the 20-aspect cap. Not an error —
    # same overload signal: the namespace likely needs splitting into sub-namespaces.
    gold_near_limit: bool = False


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
    # Set for single-chunk items (Silver/Bronze/linked); None for aggregated
    # Gold items, which are rendered from several Gold chunks at once.
    chunk_id: str | None = None


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
    # v1.8: chunk_ids of items that were skipped because the budget was full.
    # Gold-aggregated items (which span several Gold chunks) cannot map to a
    # single chunk_id and are not included here — `omitted_items` remains the
    # authoritative count. The agent can use these ids with `list_chunks` to
    # pull just the dropped evidence without a second `read_context` call.
    omitted_chunk_ids: list[str] = field(default_factory=list)

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
    # v1.7: routing match provenance. "gold" — top score came from a Gold aspect
    # (or path-token fallback for nodes without Gold); "content_fallback" — the
    # namespace was promoted by a Bronze/Silver chunk match when no Gold candidate
    # cleared the fallback_threshold. Defaults preserve v1.5/v1.6 behaviour.
    match_source: str = "gold"
