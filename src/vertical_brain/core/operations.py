from __future__ import annotations

import re
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING, Any, get_args

from vertical_brain.core.errors import (
    ChunkNotFoundError,
    DuplicateBronzeError,
    GoldGroundingError,
    ImmutableChunkError,
    SilverConflictError,
    SilverUpdateError,
    ValidationError,
)
from vertical_brain.core.gold import (
    MAX_GOLD_ASPECTS,
    GoldAspect,
    parse_gold_aspects,
    serialize_gold_aspects,
)
from vertical_brain.core.search import lexical_search
from vertical_brain.core.models import (
    Chunk,
    ChunkInput,
    ContentType,
    Layer,
    Link,
    LinkInput,
    OperationBatchResult,
    OperationType,
    OperationResult,
    OptimisticLockException,
    RouteDecision,
    StaleCandidateInput,
    StorageOperation,
    StorageOperationBatch,
    ValidationIssue,
    ValidationResult,
    utc_now,
)
if TYPE_CHECKING:
    from vertical_brain.storage.protocol import StorageProvider


VALID_CONTENT_TYPES = set(get_args(ContentType))
VALID_LAYERS = set(Layer)
VALID_OPERATIONS = set(OperationType)

# Minimum lexical score to surface a chunk as a Bronze near-duplicate warning.
# The lexical scorer gives 2 pts per term hit in content and 3 pts per term hit in path.
# Score 8 requires multiple meaningful content-word matches — single stop-word or
# path-component overlaps (e.g. "data" matching "DataArt") are filtered out.
_SIMILAR_BRONZE_MIN_SCORE = 8

# Bronze chunks larger than this many characters produce a chunk_too_large warning.
# A well-formed "one fact per chunk" Bronze entry is typically 100–400 chars.
# Beyond 600 chars the chunk likely bundles multiple facts and its embedding will
# be diluted across unrelated topics, degrading routing and search quality.
_BRONZE_MAX_CHARS = 600

# Gold aspects are routing anchors, not summaries. Longer aspects still get written,
# but the warning tells agents to split them into shorter search tags.
_GOLD_ASPECT_MAX_CHARS = 150


def _next_overflow_path(path: str) -> str:
    parent, _, name = path.rpartition("/")
    m = re.match(r"^(.+?)_(\d+)$", name)
    suffix = f"{m.group(1)}_{int(m.group(2)) + 1}" if m else f"{name}_2"
    return f"{parent}/{suffix}" if parent else suffix


class StorageOperationExecutor:
    def __init__(self, store: "StorageProvider"):
        self.store = store

    def apply(self, operation: StorageOperation) -> OperationResult:
        validation = self.validate(operation)
        if not validation.valid:
            raise ValidationError(self._format_validation_errors(validation))

        result = self._apply_validated(operation)
        self._log_audit(operation, result)
        return result

    def apply_batch(self, batch: StorageOperationBatch) -> OperationBatchResult:
        validation = self.validate_batch(batch)
        if not validation.valid:
            raise ValidationError(self._format_validation_errors(validation))

        transaction = getattr(self.store, "transaction", None)
        context = transaction() if callable(transaction) else nullcontext()
        with context:
            if batch.branch_path is not None and batch.start_version is not None:
                current_node = self.store.get_node(batch.branch_path)
                if current_node is not None and current_node.version != batch.start_version:
                    raise OptimisticLockException(
                        f"Optimistic lock conflict on '{batch.branch_path}': "
                        f"expected version {batch.start_version}, found {current_node.version}."
                    )
            results = []
            for operation in batch.operations:
                if not operation.reasoning_summary and batch.reasoning_summary:
                    operation = dc_replace(operation, reasoning_summary=batch.reasoning_summary)
                result = self._apply_validated(operation)
                self._log_audit(operation, result)
                results.append(result)

        return OperationBatchResult(
            results=results,
            status="applied",
            validation=validation,
        )

    def _log_audit(self, operation: StorageOperation, result: OperationResult) -> None:
        log_audit = getattr(self.store, "log_audit", None)
        if callable(log_audit):
            log_audit(operation, result)

    def dry_run(self, operation: StorageOperation) -> OperationResult:
        validation = self.validate(operation)
        return OperationResult(
            operation=operation.operation,
            target_path=operation.target_path,
            stale_candidates=operation.stale_candidates,
            status="dry_run" if validation.valid else "invalid",
            validation=validation,
        )

    def dry_run_batch(self, batch: StorageOperationBatch) -> OperationBatchResult:
        validation = self.validate_batch(batch)
        if not validation.valid:
            return OperationBatchResult(status="invalid", validation=validation)

        return OperationBatchResult(
            results=[self.dry_run(operation) for operation in batch.operations],
            status="dry_run",
            validation=validation,
        )

    def validate(self, operation: StorageOperation) -> ValidationResult:
        issues: list[ValidationIssue] = []
        self._validate_operation(operation, path="operation", issues=issues)
        return ValidationResult(valid=not issues, issues=issues)

    def validate_batch(self, batch: StorageOperationBatch) -> ValidationResult:
        issues: list[ValidationIssue] = []
        if (batch.branch_path is None) != (batch.start_version is None):
            issues.append(
                ValidationIssue(
                    path="batch",
                    message="branch_path and start_version must be provided together",
                )
            )
        if batch.branch_path is not None:
            self._validate_namespace_path(batch.branch_path, path="batch.branch_path", issues=issues)
        if batch.start_version is not None:
            if not isinstance(batch.start_version, int) or isinstance(batch.start_version, bool):
                issues.append(
                    ValidationIssue(
                        path="batch.start_version",
                        message="start_version must be a non-negative integer",
                    )
                )
            elif batch.start_version < 0:
                issues.append(
                    ValidationIssue(
                        path="batch.start_version",
                        message="start_version must be a non-negative integer",
                    )
                )
        for index, operation in enumerate(batch.operations):
            self._validate_operation(operation, path=f"operations[{index}]", issues=issues)
        return ValidationResult(valid=not issues, issues=issues)

    def _apply_validated(self, operation: StorageOperation) -> OperationResult:
        handler = self._APPLY_DISPATCH.get(operation.operation)
        if handler is None:
            raise ValueError(f"Unsupported storage operation: {operation.operation}")
        return handler(self, operation)

    def _op_create_node(self, operation: StorageOperation) -> OperationResult:
        self.store.ensure_node(operation.target_path)
        return OperationResult(operation=operation.operation, target_path=operation.target_path)

    def _op_append_chunk(self, operation: StorageOperation) -> OperationResult:
        assert operation.chunk is not None
        candidate = self._chunk_from_input(operation.target_path, operation.chunk)
        is_bronze = candidate.layer == "bronze"
        is_immutable = getattr(candidate, "immutable", False)

        # Exact-hash dedup: skip for immutable (silently allow duplicate)
        if is_bronze and not is_immutable and self.store.has_active_chunk_with_hash(
            operation.target_path, candidate.content_hash
        ):
            raise DuplicateBronzeError(
                f"Bronze chunk with identical content already exists at '{operation.target_path}'. "
                f"Do not write duplicates — Silver stays clean when Bronze is unique. "
                f"If the fact changed, mark the old chunk stale first with "
                f"mark_stale('{operation.target_path}', [<chunk_id>]), then write the updated content."
            )
        if candidate.layer == "silver" and candidate.source != "optimizer:namespace_compaction":
            existing_silver = [
                c for c in self.store.get_chunks_by_path(operation.target_path, include_children=False)
                if c.layer == "silver" and c.status == "active"
            ]
            if existing_silver:
                raise SilverConflictError(
                    f"append_chunk(layer=silver) rejected at '{operation.target_path}': "
                    f"an active Silver already exists (id: {existing_silver[0].id}). "
                    f"Use update_silver with current_silver_id='{existing_silver[0].id}' to replace it. "
                    f"append_chunk(layer=silver) is only for creating the first Silver summary."
                )
        chunk = self.store.save_chunk(candidate)
        links = [
            self.store.save_link(self._link_from_input(operation.target_path, link_input))
            for link_input in operation.links
        ]
        self._mark_ancestors_dirty(operation.target_path)
        # Similar-bronze and chunk_too_large: skip for immutable chunks
        similar = (
            self._find_similar_bronze(operation.target_path, candidate.content, exclude_id=chunk.id)
            if is_bronze and not is_immutable
            else []
        )
        too_large = is_bronze and not is_immutable and len(candidate.content) > _BRONZE_MAX_CHARS
        return OperationResult(
            operation=operation.operation,
            target_path=operation.target_path,
            chunk_id=chunk.id,
            link_ids=[link.id for link in links],
            stale_candidates=operation.stale_candidates,
            similar_bronze=similar,
            chunk_too_large=too_large,
        )

    def _op_create_link(self, operation: StorageOperation) -> OperationResult:
        links = [
            self.store.save_link(self._link_from_input(operation.target_path, link_input))
            for link_input in operation.links
        ]
        self._mark_ancestors_dirty(operation.target_path)
        for link_input in operation.links:
            self._mark_ancestors_dirty(link_input.target_path)
        return OperationResult(
            operation=operation.operation,
            target_path=operation.target_path,
            link_ids=[link.id for link in links],
        )

    def _op_mark_stale(self, operation: StorageOperation) -> OperationResult:
        self._update_chunk_status(operation, "stale")
        self._mark_ancestors_dirty(operation.target_path)
        return OperationResult(operation=operation.operation, target_path=operation.target_path)

    def _op_supersede_chunk(self, operation: StorageOperation) -> OperationResult:
        self._update_chunk_status(operation, "superseded")
        self._mark_ancestors_dirty(operation.target_path)
        return OperationResult(operation=operation.operation, target_path=operation.target_path)

    def _op_append_gold_aspect(self, operation: StorageOperation) -> OperationResult:
        assert operation.gold_aspect is not None
        overflow_path = self._apply_append_gold_aspect(operation.target_path, operation.gold_aspect)
        aspect_too_long = len(operation.gold_aspect.strip()) > _GOLD_ASPECT_MAX_CHARS
        self._mark_ancestors_dirty(operation.target_path)
        if overflow_path is not None:
            self._mark_ancestors_dirty(overflow_path)
        return OperationResult(
            operation=operation.operation,
            target_path=operation.target_path,
            overflow_path=overflow_path,
            aspect_too_long=aspect_too_long,
        )

    def _op_rename_namespace(self, operation: StorageOperation) -> OperationResult:
        assert operation.new_path is not None
        self._do_rename_namespace(operation.target_path, operation.new_path)
        self._mark_ancestors_dirty(operation.new_path)
        return OperationResult(operation=operation.operation, target_path=operation.target_path)

    def _op_update_silver(self, operation: StorageOperation) -> OperationResult:
        assert operation.chunk is not None
        assert operation.current_silver_id is not None

        # Runtime re-check — storage state may have changed since validation
        get_chunk = getattr(self.store, "get_chunk", None)
        old_chunk = get_chunk(operation.current_silver_id) if callable(get_chunk) else next(
            (c for c in self.store.list_chunks() if c.id == operation.current_silver_id), None
        )
        if old_chunk is None:
            raise ChunkNotFoundError(
                f"update_silver rejected: chunk '{operation.current_silver_id}' not found. "
                f"Call read_context('{operation.target_path}') to get the current Silver id."
            )
        if old_chunk.status != "active":
            raise SilverUpdateError(
                f"update_silver rejected: chunk '{operation.current_silver_id}' is {old_chunk.status}, not active. "
                f"Call read_context('{operation.target_path}') to get the current Silver id."
            )
        if old_chunk.layer != "silver":
            raise SilverUpdateError(
                f"update_silver rejected: chunk '{operation.current_silver_id}' is layer={old_chunk.layer}, not silver."
            )
        if old_chunk.node_path != operation.target_path:
            raise SilverUpdateError(
                f"update_silver rejected: chunk '{operation.current_silver_id}' belongs to "
                f"'{old_chunk.node_path}', not '{operation.target_path}'."
            )

        # Validate source_chunk_ids — storage-level checks
        for src_id in operation.source_chunk_ids:
            src = get_chunk(src_id) if callable(get_chunk) else next(
                (c for c in self.store.list_chunks() if c.id == src_id), None
            )
            if src is None:
                raise ChunkNotFoundError(f"update_silver rejected: source chunk '{src_id}' not found.")
            if src.layer == "gold":
                raise SilverUpdateError(
                    f"update_silver rejected: source chunk '{src_id}' is Gold. "
                    f"Gold cannot be used as source evidence for Silver."
                )

        # Build new Silver chunk
        new_chunk = self._chunk_from_input(operation.target_path, operation.chunk)
        new_chunk.layer = "silver"
        new_chunk.lineage = [operation.current_silver_id] + list(operation.source_chunk_ids)
        new_chunk.supersedes = [operation.current_silver_id]
        new_chunk = self.store.save_chunk(new_chunk)

        # Supersede old Silver
        old_chunk.status = "superseded"
        old_chunk.valid_to = utc_now()
        old_chunk.updated_at = utc_now()
        self.store.update_chunk(old_chunk)

        self._mark_ancestors_dirty(operation.target_path)
        return OperationResult(
            operation=operation.operation,
            target_path=operation.target_path,
            chunk_id=new_chunk.id,
        )

    _APPLY_DISPATCH = {
        "create_node": _op_create_node,
        "append_chunk": _op_append_chunk,
        "create_link": _op_create_link,
        "mark_stale": _op_mark_stale,
        "supersede_chunk": _op_supersede_chunk,
        "append_gold_aspect": _op_append_gold_aspect,
        "rename_namespace": _op_rename_namespace,
        "update_silver": _op_update_silver,
    }

    def _apply_append_gold_aspect(self, path: str, aspect: str) -> str | None:
        """Returns overflow path if a new sibling was created, otherwise None."""
        # Gold requires Silver to exist — enforce the layering contract.
        all_chunks = self.store.get_chunks_by_path(path, include_children=False)
        active_silver = [c for c in all_chunks if c.layer == "silver" and c.status == "active"]
        if not active_silver:
            raise GoldGroundingError(
                f"append_gold_aspect rejected for '{path}': no active Silver chunk found. "
                f"Gold must be grounded in Silver. "
                f"Call update_silver('{path}', ...) first to write a Silver summary, "
                f"then promote a stable conclusion to Gold."
            )
        tail = self._gold_tail_path(path)
        gold_chunks = [
            c for c in self.store.get_chunks_by_path(tail)
            if c.layer == "gold" and c.status == "active"
        ]
        if not gold_chunks:
            self.store.ensure_node(tail)
            self.store.save_chunk(
                Chunk(node_path=tail, content=serialize_gold_aspects([GoldAspect(text=aspect)]),
                      layer="gold", source="model")
            )
            return None

        latest = max(gold_chunks, key=lambda c: c.created_at)
        aspects = parse_gold_aspects(latest.content)

        # Dedup: refresh updated_at if exact text match already exists.
        for existing in aspects:
            if existing.text == aspect:
                existing.updated_at = utc_now()
                latest.content = serialize_gold_aspects(aspects)
                latest.updated_at = utc_now()
                self.store.update_chunk(latest)
                return None

        if len(aspects) < MAX_GOLD_ASPECTS:
            aspects.append(GoldAspect(text=aspect))
            latest.status = "superseded"
            latest.updated_at = utc_now()
            self.store.update_chunk(latest)
            self.store.save_chunk(
                Chunk(node_path=tail, content=serialize_gold_aspects(aspects),
                      layer="gold", source="model", lineage=[latest.id])
            )
            return None
        else:
            overflow = _next_overflow_path(tail)
            self.store.ensure_node(overflow)
            self.store.save_chunk(
                Chunk(node_path=overflow,
                      content=serialize_gold_aspects([GoldAspect(text=aspect)]),
                      layer="gold", source="model")
            )
            self.store.save_link(Link(
                source_path=tail,
                target_path=overflow,
                link_type="gold_overflow",
                reason="Gold capacity overflow.",
            ))
            return overflow

    def _gold_tail_path(self, path: str) -> str:
        current = path
        visited: set[str] = {path}
        while True:
            links = [
                lnk for lnk in self.store.get_links_by_source(current)
                if lnk.link_type == "gold_overflow"
            ]
            if not links:
                return current
            nxt = links[0].target_path
            if nxt in visited:
                return current
            visited.add(nxt)
            current = nxt

    def _chunk_from_input(self, target_path: str, chunk_input: ChunkInput) -> Chunk:
        return Chunk(
            node_path=target_path,
            content=chunk_input.content,
            layer=chunk_input.layer,
            content_type=chunk_input.content_type,
            source=chunk_input.source,
            confidence=chunk_input.confidence,
            lineage=chunk_input.lineage,
            immutable=getattr(chunk_input, "immutable", False),
        )

    def _find_similar_bronze(
        self, node_path: str, content: str, *, exclude_id: str, limit: int = 3
    ) -> list:
        """Return up to *limit* active Bronze chunks at *node_path* that are lexically
        similar to *content*, excluding the chunk just written (*exclude_id*).

        Uses the store's native FTS when available (SQLiteStore/FTS5); falls back to
        in-memory lexical scan for JsonStore.  Either way it touches only the indexed
        rows for that specific path — O(log n) in the happy path.
        """
        from vertical_brain.core.models import SearchResult  # local to avoid circular

        native_search = getattr(self.store, "search", None)
        if callable(native_search):
            # Native FTS — already indexed; filter to bronze + exact path afterwards.
            raw: list[SearchResult] = native_search(content, root_path=node_path, limit=limit * 4)
            candidates = [
                r for r in raw
                if r.layer == "bronze"
                and r.path == node_path
                and r.chunk_id != exclude_id
                and r.score >= _SIMILAR_BRONZE_MIN_SCORE
            ]
        else:
            bronze_at_path = [
                c for c in self.store.get_chunks_by_path(node_path, include_children=False)
                if c.layer == "bronze" and c.status == "active" and c.id != exclude_id
            ]
            candidates = [
                r for r in lexical_search(
                    chunks=bronze_at_path,
                    query=content,
                    root_path=node_path,
                    limit=limit * 4,
                )
                if r.score >= _SIMILAR_BRONZE_MIN_SCORE
            ]

        return candidates[:limit]

    def _link_from_input(self, source_path: str, link_input: LinkInput) -> Link:
        return Link(
            source_path=source_path,
            target_path=link_input.target_path,
            link_type=link_input.link_type,
            reason=link_input.reason,
        )

    def _do_rename_namespace(self, old_prefix: str, new_prefix: str) -> None:
        rename = getattr(self.store, "rename_namespace", None)
        if callable(rename):
            rename(old_prefix, new_prefix)
            return
        # Fallback: Python-level cascade for stores without native rename support.
        for chunk in list(self.store.list_chunks()):
            if chunk.node_path == old_prefix or chunk.node_path.startswith(old_prefix + "/"):
                chunk.node_path = new_prefix + chunk.node_path[len(old_prefix):]
                self.store.update_chunk(chunk)
        for link in list(self.store.list_links()):
            changed = False
            if link.source_path == old_prefix or link.source_path.startswith(old_prefix + "/"):
                link.source_path = new_prefix + link.source_path[len(old_prefix):]
                changed = True
            if link.target_path == old_prefix or link.target_path.startswith(old_prefix + "/"):
                link.target_path = new_prefix + link.target_path[len(old_prefix):]
                changed = True
            if changed:
                save_link = getattr(self.store, "save_link", None)
                if callable(save_link):
                    save_link(link)

    def _mark_ancestors_dirty(self, path: str) -> None:
        parts = path.split("/")
        ancestors = ["/".join(parts[:depth]) for depth in range(1, len(parts) + 1)]

        bump = getattr(self.store, "bump_nodes_dirty", None)
        if callable(bump):
            bump(ancestors)
            return

        # Fallback for stores without a batch bump: per-ancestor read + write.
        update_node = getattr(self.store, "update_node", None)
        if not callable(update_node):
            return
        for ancestor in ancestors:
            node = self.store.get_node(ancestor)
            if node is not None:
                node.is_dirty = True
                node.version += 1
                update_node(node)

    def _update_chunk_status(self, operation: StorageOperation, status: str) -> None:
        if not operation.chunk_ids:
            raise ValueError(f"{operation.operation} operation requires chunk_ids")

        for chunk_id in operation.chunk_ids:
            chunk = self.store.get_chunk(chunk_id)
            if chunk is None:
                raise ChunkNotFoundError(f"Chunk not found: {chunk_id}")
            if chunk.node_path != operation.target_path:
                raise ValueError(f"Chunk {chunk_id} does not belong to {operation.target_path}")
            if getattr(chunk, "immutable", False) and not getattr(operation, "force_immutable", False):
                raise ImmutableChunkError(
                    f"Cannot mark immutable chunk '{chunk_id}' as {status} — immutable artifacts are "
                    f"protected from modification. To replace this artifact, create a new chunk and link to it."
                )
            chunk.status = status
            chunk.updated_at = utc_now()
            if status in ("stale", "superseded", "legacy", "contradicted"):
                chunk.valid_to = utc_now()
            self.store.update_chunk(chunk)

    def _validate_operation(
        self,
        operation: StorageOperation,
        *,
        path: str,
        issues: list[ValidationIssue],
    ) -> None:
        if operation.operation not in VALID_OPERATIONS:
            issues.append(
                ValidationIssue(
                    path=f"{path}.operation",
                    message=f"Unsupported storage operation: {operation.operation}",
                )
            )
            return

        self._validate_namespace_path(operation.target_path, path=f"{path}.target_path", issues=issues)
        self._validate_confidence(operation.confidence, path=f"{path}.confidence", issues=issues)

        if operation.operation == "append_chunk":
            if operation.chunk is None:
                issues.append(ValidationIssue(path=f"{path}.chunk", message="append_chunk requires chunk"))
            else:
                self._validate_chunk_input(operation.chunk, path=f"{path}.chunk", issues=issues)
            self._validate_link_inputs(operation.links, path=f"{path}.links", issues=issues)
            self._validate_stale_candidates(
                operation.stale_candidates,
                path=f"{path}.stale_candidates",
                issues=issues,
            )
            return

        if operation.operation == "create_link":
            if not operation.links:
                issues.append(ValidationIssue(path=f"{path}.links", message="create_link requires at least one link"))
            self._validate_link_inputs(operation.links, path=f"{path}.links", issues=issues)
            return

        if operation.operation in {"mark_stale", "supersede_chunk"}:
            self._validate_chunk_ids(operation, path=f"{path}.chunk_ids", issues=issues)
            return

        if operation.operation == "append_gold_aspect":
            if not isinstance(operation.gold_aspect, str) or not operation.gold_aspect.strip():
                issues.append(
                    ValidationIssue(
                        path=f"{path}.gold_aspect",
                        message="append_gold_aspect requires non-empty gold_aspect",
                    )
                )
            return

        if operation.operation == "update_silver":
            if not operation.current_silver_id or not operation.current_silver_id.strip():
                issues.append(ValidationIssue(
                    path=f"{path}.current_silver_id",
                    message="update_silver requires a non-empty current_silver_id",
                ))
            if operation.chunk is None:
                issues.append(ValidationIssue(path=f"{path}.chunk", message="update_silver requires chunk"))
            else:
                if not isinstance(operation.chunk.content, str) or not operation.chunk.content.strip():
                    issues.append(ValidationIssue(path=f"{path}.chunk.content", message="chunk content must be non-empty"))
                if operation.chunk.layer != "silver":
                    issues.append(ValidationIssue(path=f"{path}.chunk.layer", message="update_silver chunk.layer must be 'silver'"))
            seen: set[str] = set()
            for src_id in operation.source_chunk_ids:
                if not isinstance(src_id, str) or not src_id.strip():
                    issues.append(ValidationIssue(
                        path=f"{path}.source_chunk_ids",
                        message="source_chunk_ids items must be non-empty strings",
                    ))
                elif src_id in seen:
                    issues.append(ValidationIssue(
                        path=f"{path}.source_chunk_ids",
                        message=f"Duplicate source chunk id: {src_id}",
                    ))
                else:
                    seen.add(src_id)
            return

        if operation.operation == "rename_namespace":
            if not isinstance(operation.new_path, str) or not operation.new_path.strip():
                issues.append(ValidationIssue(
                    path=f"{path}.new_path",
                    message="rename_namespace requires non-empty new_path",
                ))
                return
            self._validate_namespace_path(operation.new_path, path=f"{path}.new_path", issues=issues)
            if operation.new_path == operation.target_path:
                issues.append(ValidationIssue(
                    path=f"{path}.new_path",
                    message="rename_namespace: new_path must differ from target_path",
                ))
                return
            if operation.new_path.startswith(operation.target_path + "/"):
                issues.append(ValidationIssue(
                    path=f"{path}.new_path",
                    message="rename_namespace: cannot rename a namespace into its own descendant",
                ))
                return
            get_node = getattr(self.store, "get_node", None)
            if callable(get_node):
                if get_node(operation.target_path) is None:
                    issues.append(ValidationIssue(
                        path=f"{path}.target_path",
                        message=f"rename_namespace: target_path '{operation.target_path}' does not exist",
                    ))
                if get_node(operation.new_path) is not None:
                    issues.append(ValidationIssue(
                        path=f"{path}.new_path",
                        message=f"rename_namespace: new_path '{operation.new_path}' already exists",
                    ))
            return

    def _validate_chunk_input(self, chunk: ChunkInput, *, path: str, issues: list[ValidationIssue]) -> None:
        if not isinstance(chunk.content, str) or not chunk.content.strip():
            issues.append(ValidationIssue(path=f"{path}.content", message="chunk content must be non-empty"))
        if chunk.layer not in VALID_LAYERS:
            issues.append(ValidationIssue(path=f"{path}.layer", message=f"Unsupported layer: {chunk.layer}"))
        if chunk.content_type not in VALID_CONTENT_TYPES:
            issues.append(
                ValidationIssue(
                    path=f"{path}.content_type",
                    message=f"Unsupported content_type: {chunk.content_type}",
                )
            )
        self._validate_confidence(chunk.confidence, path=f"{path}.confidence", issues=issues)

    def _validate_link_inputs(
        self,
        links: list[LinkInput],
        *,
        path: str,
        issues: list[ValidationIssue],
    ) -> None:
        for index, link in enumerate(links):
            link_path = f"{path}[{index}]"
            self._validate_namespace_path(link.target_path, path=f"{link_path}.target_path", issues=issues)
            if not isinstance(link.link_type, str) or not link.link_type.strip():
                issues.append(ValidationIssue(path=f"{link_path}.link_type", message="link_type must be non-empty"))
            if not isinstance(link.reason, str) or not link.reason.strip():
                issues.append(ValidationIssue(path=f"{link_path}.reason", message="reason must be non-empty"))

    def _validate_stale_candidates(
        self,
        candidates: list[StaleCandidateInput],
        *,
        path: str,
        issues: list[ValidationIssue],
    ) -> None:
        for index, candidate in enumerate(candidates):
            candidate_path = f"{path}[{index}]"
            self._validate_namespace_path(candidate.path, path=f"{candidate_path}.path", issues=issues)
            if not isinstance(candidate.reason, str) or not candidate.reason.strip():
                issues.append(ValidationIssue(path=f"{candidate_path}.reason", message="reason must be non-empty"))

    def _validate_chunk_ids(
        self,
        operation: StorageOperation,
        *,
        path: str,
        issues: list[ValidationIssue],
    ) -> None:
        if not operation.chunk_ids:
            issues.append(ValidationIssue(path=path, message=f"{operation.operation} requires chunk_ids"))
            return

        seen_ids: set[str] = set()
        for index, chunk_id in enumerate(operation.chunk_ids):
            chunk_id_path = f"{path}[{index}]"
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                issues.append(ValidationIssue(path=chunk_id_path, message="chunk id must be non-empty"))
                continue
            if chunk_id in seen_ids:
                issues.append(ValidationIssue(path=chunk_id_path, message=f"Duplicate chunk id: {chunk_id}"))
                continue
            seen_ids.add(chunk_id)
            chunk = self.store.get_chunk(chunk_id)
            if chunk is None:
                issues.append(ValidationIssue(path=chunk_id_path, message=f"Chunk not found: {chunk_id}"))
                continue
            if chunk.node_path != operation.target_path:
                issues.append(
                    ValidationIssue(
                        path=chunk_id_path,
                        message=f"Chunk {chunk_id} does not belong to {operation.target_path}",
                    )
                )

    def _validate_namespace_path(self, namespace_path: object, *, path: str, issues: list[ValidationIssue]) -> None:
        if not isinstance(namespace_path, str) or not namespace_path.strip():
            issues.append(ValidationIssue(path=path, message="namespace path must be non-empty"))
            return
        if namespace_path != namespace_path.strip():
            issues.append(ValidationIssue(path=path, message="namespace path must not have surrounding whitespace"))
        if namespace_path.startswith("/") or namespace_path.endswith("/") or "//" in namespace_path:
            issues.append(ValidationIssue(path=path, message="namespace path must use non-empty slash-separated parts"))

    def _validate_confidence(self, confidence: object, *, path: str, issues: list[ValidationIssue]) -> None:
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            issues.append(ValidationIssue(path=path, message="confidence must be a number between 0 and 1"))
            return
        if not 0 <= confidence <= 1:
            issues.append(ValidationIssue(path=path, message="confidence must be between 0 and 1"))

    def _format_validation_errors(self, validation: ValidationResult) -> str:
        return "; ".join(f"{issue.path}: {issue.message}" for issue in validation.issues)


def operation_batch_from_dict(payload: Mapping[str, Any]) -> StorageOperationBatch:
    if not isinstance(payload, Mapping):
        raise ValueError("StorageOperationBatch payload must be an object")
    if "operations" not in payload:
        return StorageOperationBatch(operations=[operation_from_dict(payload)])

    unknown = set(payload) - {"operations", "reasoning_summary", "branch_path", "start_version"}
    if unknown:
        raise ValueError(f"Unknown StorageOperationBatch fields: {', '.join(sorted(unknown))}")
    operations_payload = payload["operations"]
    if not isinstance(operations_payload, list):
        raise ValueError("StorageOperationBatch.operations must be a list")
    start_version = payload.get("start_version")
    if start_version is not None and (not isinstance(start_version, int) or isinstance(start_version, bool)):
        raise ValueError("StorageOperationBatch.start_version must be an integer")
    branch_path = payload.get("branch_path")
    if branch_path is not None and not isinstance(branch_path, str):
        raise ValueError("StorageOperationBatch.branch_path must be a string")
    return StorageOperationBatch(
        operations=[operation_from_dict(operation) for operation in operations_payload],
        reasoning_summary=_string_value(payload, "reasoning_summary", default=""),
        branch_path=branch_path,
        start_version=start_version,
    )


def operation_from_dict(payload: Mapping[str, Any]) -> StorageOperation:
    if not isinstance(payload, Mapping):
        raise ValueError("StorageOperation payload must be an object")
    unknown = set(payload) - {
        "operation",
        "target_path",
        "chunk",
        "links",
        "stale_candidates",
        "chunk_ids",
        "gold_aspect",
        "new_path",
        "confidence",
        "reasoning_summary",
        "current_silver_id",
        "source_chunk_ids",
    }
    if unknown:
        raise ValueError(f"Unknown StorageOperation fields: {', '.join(sorted(unknown))}")

    chunk_payload = payload.get("chunk")
    chunk = _chunk_input_from_dict(chunk_payload) if chunk_payload is not None else None
    links_payload = _list_value(payload, "links", default=[])
    stale_payload = _list_value(payload, "stale_candidates", default=[])
    current_silver_id = payload.get("current_silver_id")
    source_chunk_ids = [_string_item(x, "source_chunk_ids") for x in _list_value(payload, "source_chunk_ids", default=[])]

    return StorageOperation(
        operation=_string_value(payload, "operation"),
        target_path=_string_value(payload, "target_path"),
        chunk=chunk,
        links=[_link_input_from_dict(link) for link in links_payload],
        stale_candidates=[_stale_candidate_from_dict(candidate) for candidate in stale_payload],
        chunk_ids=[_string_item(chunk_id, "chunk_ids") for chunk_id in _list_value(payload, "chunk_ids", default=[])],
        gold_aspect=payload.get("gold_aspect"),
        new_path=payload.get("new_path"),
        confidence=_number_value(payload, "confidence", default=1.0),
        reasoning_summary=_string_value(payload, "reasoning_summary", default=""),
        current_silver_id=current_silver_id,
        source_chunk_ids=source_chunk_ids,
    )


def _chunk_input_from_dict(payload: object) -> ChunkInput:
    if not isinstance(payload, Mapping):
        raise ValueError("StorageOperation.chunk must be an object")
    unknown = set(payload) - {"content", "layer", "content_type", "source", "confidence", "lineage", "immutable"}
    if unknown:
        raise ValueError(f"Unknown ChunkInput fields: {', '.join(sorted(unknown))}")
    immutable_val = payload.get("immutable", False)
    if not isinstance(immutable_val, bool):
        raise ValueError("immutable must be a boolean")
    return ChunkInput(
        content=_string_value(payload, "content"),
        layer=_string_value(payload, "layer", default="bronze"),
        content_type=_string_value(payload, "content_type", default="note"),
        source=_string_value(payload, "source", default="model"),
        confidence=_number_value(payload, "confidence", default=1.0),
        lineage=[_string_item(item, "lineage") for item in _list_value(payload, "lineage", default=[])],
        immutable=immutable_val,
    )


def _link_input_from_dict(payload: object) -> LinkInput:
    if not isinstance(payload, Mapping):
        raise ValueError("StorageOperation.links items must be objects")
    unknown = set(payload) - {"target_path", "link_type", "reason"}
    if unknown:
        raise ValueError(f"Unknown LinkInput fields: {', '.join(sorted(unknown))}")
    return LinkInput(
        target_path=_string_value(payload, "target_path"),
        link_type=_string_value(payload, "link_type"),
        reason=_string_value(payload, "reason"),
    )


def _stale_candidate_from_dict(payload: object) -> StaleCandidateInput:
    if not isinstance(payload, Mapping):
        raise ValueError("StorageOperation.stale_candidates items must be objects")
    unknown = set(payload) - {"path", "reason"}
    if unknown:
        raise ValueError(f"Unknown StaleCandidateInput fields: {', '.join(sorted(unknown))}")
    return StaleCandidateInput(
        path=_string_value(payload, "path"),
        reason=_string_value(payload, "reason"),
    )


def _string_value(payload: Mapping[str, Any], key: str, *, default: str | None = None) -> str:
    if key not in payload:
        if default is not None:
            return default
        raise ValueError(f"Missing required field: {key}")
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _number_value(payload: Mapping[str, Any], key: str, *, default: float) -> float:
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _list_value(payload: Mapping[str, Any], key: str, *, default: list[Any]) -> list[Any]:
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _string_item(value: object, list_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{list_name} items must be strings")
    return value


def operation_to_staging(content: str, original_target: str, reason: str) -> StorageOperation:
    """Build an operation that writes a Bronze chunk to the staging buffer.

    Used when the router cannot classify with enough confidence.  The chunk
    will remain in STAGING/Unclassified until a future optimize pass or manual
    reclassification.
    """
    from vertical_brain.core.models import STAGING_PATH
    return StorageOperation(
        operation="append_chunk",
        target_path=STAGING_PATH,
        chunk=ChunkInput(
            content=content,
            layer="bronze",
            content_type="note",
            source="staging",
            confidence=0.0,
        ),
        reasoning_summary=f"Staged from '{original_target}': {reason}",
    )


def operation_from_route_decision(decision: RouteDecision, content: str) -> StorageOperation:
    return StorageOperation(
        operation="append_chunk",
        target_path=decision.target_path,
        chunk=ChunkInput(
            content=content,
            layer=decision.layer,
            content_type=decision.content_type,
            source="model",
            confidence=decision.confidence,
        ),
        links=[
            LinkInput(
                target_path=peer.path,
                link_type="peer",
                reason=peer.reason,
            )
            for peer in decision.peer_links
        ],
        stale_candidates=[
            StaleCandidateInput(path=candidate.path, reason=candidate.reason)
            for candidate in decision.stale_candidates
        ],
        confidence=decision.confidence,
        reasoning_summary=decision.reasoning_summary,
    )
