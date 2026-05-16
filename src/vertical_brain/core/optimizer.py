from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from vertical_brain.core.models import (
    Chunk,
    ChunkInput,
    ContentType,
    StorageOperation,
    StorageOperationBatch,
)
from vertical_brain.core.operations import StorageOperationExecutor

if TYPE_CHECKING:
    from vertical_brain.storage.protocol import StorageProvider


COMPACTION_SOURCE = "optimizer:namespace_compaction"


class SimpleOptimizer:
    """Transactional branch optimizer.

    optimize_branch(path) flow:
      1. Read storage exactly once.
      2. Build the full operation matrix via _build_plan (pure, no I/O).
      3. If there is anything to apply, execute the entire batch in a single
         transactional tick via apply_batch.
      4. Build the report from the batch counters — no second read.
    """

    def __init__(self, store: "StorageProvider", min_compaction_path_parts: int) -> None:
        self.store = store
        self.min_compaction_path_parts = min_compaction_path_parts

    # ── public interface ──────────────────────────────────────────────────────

    def optimize_branch(self, path: str) -> str:
        chunks = self.store.get_chunks_by_path(path, include_children=True)  # single read
        batch = self._build_plan(chunks, path)
        if batch.operations:
            StorageOperationExecutor(self.store).apply_batch(batch)
        return self._build_report(path, chunks, batch)

    def plan_branch(self, path: str) -> StorageOperationBatch:
        """Return the operation batch optimize_branch would apply without mutating storage."""
        chunks = self.store.get_chunks_by_path(path, include_children=True)
        return self._build_plan(chunks, path)

    # ── core planning (pure over the already-fetched chunk list) ──────────────

    def _build_plan(self, chunks: list[Chunk], path: str) -> StorageOperationBatch:
        operations: list[StorageOperation] = []
        seen: dict[tuple[str, str], str] = {}
        stale_ids: set[str] = set()

        # Pass 1 — exact-duplicate detection
        for chunk in chunks:
            if chunk.status != "active":
                continue
            dedup_key = (chunk.node_path, chunk.content_hash or chunk.content)
            if dedup_key in seen:
                operations.append(StorageOperation(
                    operation="mark_stale",
                    target_path=chunk.node_path,
                    chunk_ids=[chunk.id],
                    reasoning_summary="Exact duplicate marked stale during optimize.",
                ))
                stale_ids.add(chunk.id)
            else:
                seen[dedup_key] = chunk.id

        # Pass 2 — namespace Silver compaction
        # Candidates: active, not already queued for stale, not gold,
        # not a prior compaction output, no established lineage.
        candidates = [
            c for c in chunks
            if c.status == "active"
            and c.id not in stale_ids
            and c.layer != "gold"
            and c.source != COMPACTION_SOURCE
            and not c.lineage
        ]
        grouped: dict[str, list[Chunk]] = defaultdict(list)
        for c in candidates:
            if len(c.node_path.split("/")) >= self.min_compaction_path_parts:
                grouped[c.node_path].append(c)

        for node_path, node_chunks in sorted(grouped.items()):
            if len(node_chunks) < 2:
                continue
            lineage = [c.id for c in node_chunks]
            operations.append(StorageOperation(
                operation="append_chunk",
                target_path=node_path,
                chunk=ChunkInput(
                    content=self._build_silver_compaction(node_path, node_chunks),
                    layer="silver",
                    content_type=self._dominant_content_type(node_chunks),
                    source=COMPACTION_SOURCE,
                    confidence=min(c.confidence for c in node_chunks),
                    lineage=lineage,
                ),
                confidence=min(c.confidence for c in node_chunks),
                reasoning_summary="Canonical namespace variant created from active chunk variants.",
            ))
            operations.append(StorageOperation(
                operation="supersede_chunk",
                target_path=node_path,
                chunk_ids=lineage,
                reasoning_summary="Original variants superseded by canonical namespace compaction.",
            ))

        return StorageOperationBatch(
            operations=operations,
            reasoning_summary=f"Compaction plan for {path}.",
        )

    # ── report (derived from batch + initial chunks — no storage re-read) ────

    def _build_report(
        self,
        path: str,
        initial_chunks: list[Chunk],
        batch: StorageOperationBatch,
    ) -> str:
        duplicates_by_node: dict[str, int] = defaultdict(int)
        compactions_by_node: dict[str, int] = defaultdict(int)

        for op in batch.operations:
            if op.operation == "mark_stale":
                duplicates_by_node[op.target_path] += len(op.chunk_ids or [])
            elif (
                op.operation == "append_chunk"
                and op.chunk is not None
                and op.chunk.source == COMPACTION_SOURCE
            ):
                compactions_by_node[op.target_path] += 1

        total_ops = len(batch.operations)
        if total_ops == 0:
            no_op_line = "  no optimization changes required"
        else:
            no_op_line = None

        lines = [f"Optimize report for {path}", ""]
        if no_op_line:
            lines.append(no_op_line)
        else:
            lines.append("  observed before optimization:")
            for node_path, node_chunks in sorted(self._group_by_node(initial_chunks).items()):
                active = sum(1 for c in node_chunks if c.status == "active")
                stale = sum(1 for c in node_chunks if c.status == "stale")
                superseded = sum(1 for c in node_chunks if c.status == "superseded")
                lines.append(
                    f"  - {node_path}: {len(node_chunks)} chunks "
                    f"({active} active, {stale} stale, {superseded} superseded)"
                )
            lines.append("")
            lines.append("  changes applied:")
            for node_path, node_chunks in sorted(self._group_by_node(initial_chunks).items()):
                dups = duplicates_by_node[node_path]
                comp = compactions_by_node[node_path]
                if dups or comp:
                    lines.append(
                        f"  - {node_path}: "
                        f"{dups} exact duplicates marked stale, "
                        f"{comp} namespace compactions created"
                    )

        return "\n".join(lines)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_silver_compaction(self, node_path: str, chunks: list[Chunk]) -> str:
        lines = [f"Compacted Silver summary for namespace: {node_path}.", "Facts:"]
        for c in chunks:
            lines.append(f"- {c.content}")
        lines.append("Lineage:")
        for c in chunks:
            lines.append(f"- {c.id}")
        return "\n".join(lines)

    def _dominant_content_type(self, chunks: list[Chunk]) -> ContentType:
        counts: dict[ContentType, int] = defaultdict(int)
        for c in chunks:
            counts[c.content_type] += 1
        return sorted(counts, key=lambda ct: (-counts[ct], ct))[0]

    def _group_by_node(self, chunks: list[Chunk]) -> dict[str, list[Chunk]]:
        grouped: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            grouped[c.node_path].append(c)
        return grouped
