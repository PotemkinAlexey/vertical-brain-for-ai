from __future__ import annotations

from vertical_brain.core.models import (
    Chunk,
    ChunkInput,
    Link,
    LinkInput,
    OperationResult,
    RouteDecision,
    StaleCandidateInput,
    StorageOperation,
)
from vertical_brain.storage.json_store import JsonStore


class StorageOperationExecutor:
    def __init__(self, store: JsonStore):
        self.store = store

    def apply(self, operation: StorageOperation) -> OperationResult:
        if operation.operation == "create_node":
            self.store.ensure_node(operation.target_path)
            return OperationResult(operation=operation.operation, target_path=operation.target_path)

        if operation.operation == "append_chunk":
            if operation.chunk is None:
                raise ValueError("append_chunk operation requires chunk")
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

        raise ValueError(f"Unsupported storage operation: {operation.operation}")

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
