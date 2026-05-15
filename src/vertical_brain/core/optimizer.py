from __future__ import annotations

from collections import defaultdict

from vertical_brain.core.models import Chunk, ChunkInput, ContentType, StorageOperation, StorageOperationBatch
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.storage.json_store import JsonStore


COMPACTION_SOURCE = "optimizer:namespace_compaction"


class SimpleOptimizer:
    """MVP optimizer.

    Current behavior:
    - marks exact duplicate chunks inside a branch as stale
    - compacts active chunks inside the same exact namespace into Silver chunks
    - creates or updates a deterministic Gold summary
    - returns a simple summary report
    """

    def __init__(self, store: JsonStore, min_compaction_path_parts: int):
        self.store = store
        self.min_compaction_path_parts = min_compaction_path_parts

    def optimize_branch(self, path: str) -> str:
        chunks = self.store.get_chunks_by_path(path, include_children=True)
        duplicate_count_by_path: dict[str, int] = defaultdict(int)
        seen: dict[tuple[str, str], str] = {}

        for chunk in chunks:
            if chunk.status != "active":
                continue

            key = (chunk.node_path, chunk.content)
            if key in seen:
                StorageOperationExecutor(self.store).apply(
                    StorageOperation(
                        operation="mark_stale",
                        target_path=chunk.node_path,
                        chunk_ids=[chunk.id],
                        reasoning_summary="Exact duplicate marked stale during optimize.",
                    )
                )
                duplicate_count_by_path[chunk.node_path] += 1
            else:
                seen[key] = chunk.id

        compaction_count_by_path = self._compact_related_chunks(path)

        updated_chunks = self.store.get_chunks_by_path(path, include_children=True)

        lines = [f"Optimize report for {path}", ""]
        for node_path, node_chunks in sorted(self._group_by_node(updated_chunks).items()):
            active_count = sum(1 for chunk in node_chunks if chunk.status == "active")
            stale_count = sum(1 for chunk in node_chunks if chunk.status == "stale")
            superseded_count = sum(1 for chunk in node_chunks if chunk.status == "superseded")
            duplicates_marked = duplicate_count_by_path[node_path]
            compactions_created = compaction_count_by_path[node_path]
            lines.append(
                f"- {node_path}: {len(node_chunks)} chunks, "
                f"{active_count} active, {stale_count} stale, {superseded_count} superseded, "
                f"{duplicates_marked} exact duplicates marked stale, "
                f"{compactions_created} namespace compactions created"
            )

        return "\n".join(lines)

    def _compact_related_chunks(self, path: str) -> dict[str, int]:
        active_chunks = [
            chunk
            for chunk in self.store.get_chunks_by_path(path, include_children=True)
            if chunk.status == "active"
            and chunk.layer != "gold"
            and chunk.source != COMPACTION_SOURCE
            and not chunk.lineage
        ]
        grouped: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in active_chunks:
            if len(chunk.node_path.split("/")) < self.min_compaction_path_parts:
                continue
            grouped[chunk.node_path].append(chunk)

        compaction_count_by_path: dict[str, int] = defaultdict(int)
        for node_path, chunks in sorted(grouped.items()):
            if len(chunks) < 2:
                continue

            lineage = [chunk.id for chunk in chunks]
            batch = StorageOperationBatch(
                operations=[
                    StorageOperation(
                        operation="append_chunk",
                        target_path=node_path,
                        chunk=ChunkInput(
                            content=self._build_silver_compaction(node_path, chunks),
                            layer="silver",
                            content_type=self._dominant_content_type(chunks),
                            source=COMPACTION_SOURCE,
                            confidence=min(chunk.confidence for chunk in chunks),
                            lineage=lineage,
                        ),
                        confidence=min(chunk.confidence for chunk in chunks),
                        reasoning_summary="Canonical namespace variant created from active chunk variants.",
                    ),
                    StorageOperation(
                        operation="supersede_chunk",
                        target_path=node_path,
                        chunk_ids=lineage,
                        reasoning_summary="Original variants superseded by canonical namespace compaction.",
                    ),
                ],
                reasoning_summary="Many active variants collapsed into one canonical Silver chunk.",
            )
            StorageOperationExecutor(self.store).apply_batch(batch)

            compaction_count_by_path[node_path] += 1
        return compaction_count_by_path

    def _build_silver_compaction(self, node_path: str, chunks: list[Chunk]) -> str:
        lines = [
            f"Compacted Silver summary for namespace: {node_path}.",
            "Facts:",
        ]
        for chunk in chunks:
            lines.append(f"- {chunk.content}")
        lines.append("Lineage:")
        for chunk in chunks:
            lines.append(f"- {chunk.id}")
        return "\n".join(lines)

    def _dominant_content_type(self, chunks: list[Chunk]) -> ContentType:
        counts: dict[ContentType, int] = defaultdict(int)
        for chunk in chunks:
            counts[chunk.content_type] += 1
        return sorted(counts, key=lambda content_type: (-counts[content_type], content_type))[0]

    def _group_by_node(self, chunks: list[Chunk]) -> dict[str, list[Chunk]]:
        grouped: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.node_path].append(chunk)
        return grouped

