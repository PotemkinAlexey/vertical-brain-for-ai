from __future__ import annotations

import re
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING, Any, get_args

from vertical_brain.core.gold import parse_gold_content
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
VALID_LAYERS = set(get_args(Layer))
VALID_OPERATIONS = set(get_args(OperationType))
MAX_GOLD_CHARS = 200


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
            raise ValueError(self._format_validation_errors(validation))

        result = self._apply_validated(operation)
        self._log_audit(operation, result)
        return result

    def apply_batch(self, batch: StorageOperationBatch) -> OperationBatchResult:
        validation = self.validate_batch(batch)
        if not validation.valid:
            raise ValueError(self._format_validation_errors(validation))

        transaction = getattr(self.store, "transaction", None)
        context = transaction() if callable(transaction) else nullcontext()
        with context:
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
        for index, operation in enumerate(batch.operations):
            self._validate_operation(operation, path=f"operations[{index}]", issues=issues)
        return ValidationResult(valid=not issues, issues=issues)

    def _apply_validated(self, operation: StorageOperation) -> OperationResult:
        if operation.operation == "create_node":
            self.store.ensure_node(operation.target_path)
            return OperationResult(operation=operation.operation, target_path=operation.target_path)

        if operation.operation == "append_chunk":
            assert operation.chunk is not None
            chunk = self.store.save_chunk(self._chunk_from_input(operation.target_path, operation.chunk))
            links = [
                self.store.save_link(self._link_from_input(operation.target_path, link_input))
                for link_input in operation.links
            ]
            return OperationResult(
                operation=operation.operation,
                target_path=operation.target_path,
                chunk_id=chunk.id,
                link_ids=[link.id for link in links],
                stale_candidates=operation.stale_candidates,
            )

        if operation.operation == "create_link":
            links = [
                self.store.save_link(self._link_from_input(operation.target_path, link_input))
                for link_input in operation.links
            ]
            return OperationResult(
                operation=operation.operation,
                target_path=operation.target_path,
                link_ids=[link.id for link in links],
            )

        if operation.operation == "mark_stale":
            self._update_chunk_status(operation, "stale")
            return OperationResult(operation=operation.operation, target_path=operation.target_path)

        if operation.operation == "supersede_chunk":
            self._update_chunk_status(operation, "superseded")
            return OperationResult(operation=operation.operation, target_path=operation.target_path)

        if operation.operation == "append_gold_aspect":
            assert operation.gold_aspect is not None
            overflow_path = self._apply_append_gold_aspect(operation.target_path, operation.gold_aspect)
            return OperationResult(
                operation=operation.operation,
                target_path=operation.target_path,
                overflow_path=overflow_path,
            )

        raise ValueError(f"Unsupported storage operation: {operation.operation}")

    def _apply_append_gold_aspect(self, path: str, aspect: str) -> str | None:
        """Returns overflow path if a new sibling was created, otherwise None."""
        tail = self._gold_tail_path(path)
        gold_chunks = [
            c for c in self.store.get_chunks_by_path(tail)
            if c.layer == "gold" and c.status == "active"
        ]
        if not gold_chunks:
            self.store.ensure_node(tail)
            self.store.save_chunk(Chunk(node_path=tail, content=aspect, layer="gold", source="model"))
            return None

        latest = max(gold_chunks, key=lambda c: c.created_at)
        existing_aspects = parse_gold_content(latest.content)
        new_content = " | ".join(existing_aspects + [aspect])
        if len(new_content) <= MAX_GOLD_CHARS:
            latest.status = "superseded"
            latest.updated_at = utc_now()
            self.store.update_chunk(latest)
            self.store.save_chunk(
                Chunk(node_path=tail, content=new_content, layer="gold", source="model", lineage=[latest.id])
            )
            return None
        else:
            overflow = _next_overflow_path(tail)
            self.store.ensure_node(overflow)
            self.store.save_chunk(Chunk(node_path=overflow, content=aspect, layer="gold", source="model"))
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
                lnk for lnk in self.store.list_links()
                if lnk.source_path == current and lnk.link_type == "gold_overflow"
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
        )

    def _link_from_input(self, source_path: str, link_input: LinkInput) -> Link:
        return Link(
            source_path=source_path,
            target_path=link_input.target_path,
            link_type=link_input.link_type,
            reason=link_input.reason,
        )

    def _update_chunk_status(self, operation: StorageOperation, status: str) -> None:
        if not operation.chunk_ids:
            raise ValueError(f"{operation.operation} operation requires chunk_ids")

        chunks_by_id = {chunk.id: chunk for chunk in self.store.list_chunks()}
        for chunk_id in operation.chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                raise ValueError(f"Chunk not found: {chunk_id}")
            if chunk.node_path != operation.target_path:
                raise ValueError(f"Chunk {chunk_id} does not belong to {operation.target_path}")
            chunk.status = status
            chunk.updated_at = utc_now()
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

        chunks_by_id = {chunk.id: chunk for chunk in self.store.list_chunks()}
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
            chunk = chunks_by_id.get(chunk_id)
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

    unknown = set(payload) - {"operations", "reasoning_summary"}
    if unknown:
        raise ValueError(f"Unknown StorageOperationBatch fields: {', '.join(sorted(unknown))}")
    operations_payload = payload["operations"]
    if not isinstance(operations_payload, list):
        raise ValueError("StorageOperationBatch.operations must be a list")
    return StorageOperationBatch(
        operations=[operation_from_dict(operation) for operation in operations_payload],
        reasoning_summary=_string_value(payload, "reasoning_summary", default=""),
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
        "confidence",
        "reasoning_summary",
    }
    if unknown:
        raise ValueError(f"Unknown StorageOperation fields: {', '.join(sorted(unknown))}")

    chunk_payload = payload.get("chunk")
    chunk = _chunk_input_from_dict(chunk_payload) if chunk_payload is not None else None
    links_payload = _list_value(payload, "links", default=[])
    stale_payload = _list_value(payload, "stale_candidates", default=[])

    return StorageOperation(
        operation=_string_value(payload, "operation"),
        target_path=_string_value(payload, "target_path"),
        chunk=chunk,
        links=[_link_input_from_dict(link) for link in links_payload],
        stale_candidates=[_stale_candidate_from_dict(candidate) for candidate in stale_payload],
        chunk_ids=[_string_item(chunk_id, "chunk_ids") for chunk_id in _list_value(payload, "chunk_ids", default=[])],
        gold_aspect=payload.get("gold_aspect"),
        confidence=_number_value(payload, "confidence", default=1.0),
        reasoning_summary=_string_value(payload, "reasoning_summary", default=""),
    )


def _chunk_input_from_dict(payload: object) -> ChunkInput:
    if not isinstance(payload, Mapping):
        raise ValueError("StorageOperation.chunk must be an object")
    unknown = set(payload) - {"content", "layer", "content_type", "source", "confidence", "lineage"}
    if unknown:
        raise ValueError(f"Unknown ChunkInput fields: {', '.join(sorted(unknown))}")
    return ChunkInput(
        content=_string_value(payload, "content"),
        layer=_string_value(payload, "layer", default="bronze"),
        content_type=_string_value(payload, "content_type", default="note"),
        source=_string_value(payload, "source", default="model"),
        confidence=_number_value(payload, "confidence", default=1.0),
        lineage=[_string_item(item, "lineage") for item in _list_value(payload, "lineage", default=[])],
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
