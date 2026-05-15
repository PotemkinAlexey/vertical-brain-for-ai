from __future__ import annotations

from collections import defaultdict

from vertical_brain.core.models import Chunk, utc_now
from vertical_brain.storage.json_store import JsonStore


class SimpleOptimizer:
    """MVP optimizer.

    Current behavior:
    - marks exact duplicate chunks inside a branch as stale
    - creates or updates a deterministic Gold summary placeholder
    - returns a simple summary report
    - semantic compaction is intentionally left as TODO
    """

    def __init__(self, store: JsonStore):
        self.store = store

    def optimize_branch(self, path: str) -> str:
        chunks = self.store.get_chunks_by_path(path, include_children=True)
        duplicate_count_by_path: dict[str, int] = defaultdict(int)
        seen: dict[tuple[str, str], str] = {}

        for chunk in chunks:
            if chunk.status != "active":
                continue

            key = (chunk.node_path, chunk.content)
            if key in seen:
                chunk.status = "stale"
                chunk.updated_at = utc_now()
                self.store.update_chunk(chunk)
                duplicate_count_by_path[chunk.node_path] += 1
            else:
                seen[key] = chunk.id

        updated_chunks = self.store.get_chunks_by_path(path, include_children=True)
        self.store.update_node_gold_summary(path, self._build_gold_summary(path, updated_chunks))

        lines = [f"Optimize report for {path}", ""]
        for node_path, node_chunks in sorted(self._group_by_node(updated_chunks).items()):
            active_count = sum(1 for chunk in node_chunks if chunk.status == "active")
            stale_count = sum(1 for chunk in node_chunks if chunk.status == "stale")
            duplicates_marked = duplicate_count_by_path[node_path]
            lines.append(
                f"- {node_path}: {len(node_chunks)} chunks, "
                f"{active_count} active, {stale_count} stale, "
                f"{duplicates_marked} exact duplicates marked stale"
            )

        lines.append("")
        lines.append("Gold summary placeholder updated.")
        lines.append("TODO: semantic compaction.")
        return "\n".join(lines)

    def _group_by_node(self, chunks: list[Chunk]) -> dict[str, list[Chunk]]:
        grouped: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.node_path].append(chunk)
        return grouped

    def _build_gold_summary(self, path: str, chunks: list[Chunk]) -> str:
        active_chunks = [chunk for chunk in chunks if chunk.status == "active"]
        if not active_chunks:
            return f"Gold summary placeholder for {path}.\nNo active chunks under this branch yet."

        grouped = self._group_by_node(active_chunks)
        lines = [
            f"Gold summary placeholder for {path}.",
            f"Active chunks: {len(active_chunks)}.",
            "Covered nodes:",
        ]
        for node_path, node_chunks in sorted(grouped.items()):
            lines.append(f"- {node_path}: {len(node_chunks)} active chunks")
        return "\n".join(lines)
