from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from vertical_brain.core.models import (
    Chunk,
    ChunkInput,
    ContentType,
    LinkInput,
    StorageOperation,
    StorageOperationBatch,
)
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.llm.embedding import EmbeddingProvider, cosine_similarity

if TYPE_CHECKING:
    from vertical_brain.storage.protocol import StorageProvider


COMPACTION_SOURCE = "optimizer:namespace_compaction"
LINK_DISCOVERY_SOURCE = "optimizer:link_discovery"


class SimpleOptimizer:
    """Transactional branch optimizer.

    optimize_branch(path) flow:
      1. Read storage exactly once.
      2. Build the full operation matrix via _build_plan (pure, no I/O).
      3. If there is anything to apply, execute the entire batch in a single
         transactional tick via apply_batch.
      4. Build the report from the batch counters — no second read.
    """

    def __init__(
        self,
        store: "StorageProvider",
        min_compaction_path_parts: int,
        decay_rate: float = 1.0,
        decay_days: int = 30,
        stale_threshold: float = 0.2,
        embedding_provider: EmbeddingProvider | None = None,
        link_similarity_threshold: float = 0.80,
    ) -> None:
        self.store = store
        self.min_compaction_path_parts = min_compaction_path_parts
        self.decay_rate = decay_rate
        self.decay_days = decay_days
        self.stale_threshold = stale_threshold
        self._embedding_provider = embedding_provider
        self._link_similarity_threshold = link_similarity_threshold

    # ── public interface ──────────────────────────────────────────────────────

    def optimize_branch(self, path: str) -> str:
        chunks, linked_paths, node_version = self._snapshot_branch(path)
        batch = self._build_plan(chunks, path, linked_paths=linked_paths)
        if node_version is not None:
            batch.branch_path = path
            batch.start_version = node_version
        if batch.operations:
            StorageOperationExecutor(self.store).apply_batch(batch)
        return self._build_report(path, chunks, batch)

    def plan_branch(self, path: str) -> StorageOperationBatch:
        """Return the operation batch optimize_branch would apply without mutating storage."""
        chunks, linked_paths, node_version = self._snapshot_branch(path)
        batch = self._build_plan(chunks, path, linked_paths=linked_paths)
        if node_version is not None:
            batch.branch_path = path
            batch.start_version = node_version
        return batch

    def optimize_all(self) -> str:
        """Run full optimization across every namespace, then discover cross-namespace links.

        Unlike optimize_branch, this reads all chunks in one pass and applies a single
        transactional batch. OCC is not applied — this is intended as a periodic maintenance
        sweep, not a targeted branch update.
        """
        all_chunks = self.store.list_chunks()
        linked_paths = self._linked_paths_if_decay_enabled()
        batch = self._build_plan(all_chunks, "", linked_paths=linked_paths)
        if batch.operations:
            StorageOperationExecutor(self.store).apply_batch(batch)
        branch_report = self._build_report("(all namespaces)", all_chunks, batch)
        link_report = self.discover_links()
        return branch_report + "\n\n" + link_report

    def discover_links(self, root_path: str | None = None) -> str:
        """Find and create links between semantically similar Silver chunks from different namespaces.

        Requires an embedding_provider to be set at construction time.
        Reads storage once, embeds Silver chunks, then runs pure pairwise similarity.
        Creates at most one link per namespace pair, skipping pairs that are already linked.
        """
        if self._embedding_provider is None:
            return "Link discovery skipped: no embedding provider configured."
        silver_chunks, existing_pairs = self._snapshot_for_links(root_path)
        if len(silver_chunks) < 2:
            return "Link discovery: fewer than 2 Silver chunks found, nothing to link."
        vectors = self._embed_chunks(silver_chunks)
        operations = self._build_link_plan(silver_chunks, vectors, existing_pairs)
        if operations:
            batch = StorageOperationBatch(
                operations=operations,
                reasoning_summary="Cross-namespace Silver link discovery.",
            )
            StorageOperationExecutor(self.store).apply_batch(batch)
        return self._build_link_report(operations)

    def _snapshot_for_links(
        self, root_path: str | None
    ) -> tuple[list[Chunk], set[tuple[str, str]]]:
        if root_path:
            all_chunks = self.store.get_chunks_by_path(root_path, include_children=True)
        else:
            all_chunks = self.store.list_chunks()
        silver_chunks = [
            c for c in all_chunks
            if c.status == "active" and c.layer == "silver"
        ]
        existing_pairs: set[tuple[str, str]] = set()
        for lnk in self.store.list_links():
            existing_pairs.add((lnk.source_path, lnk.target_path))
            existing_pairs.add((lnk.target_path, lnk.source_path))
        return silver_chunks, existing_pairs

    def _embed_chunks(self, chunks: list[Chunk]) -> dict[str, list[float]]:
        assert self._embedding_provider is not None
        model_name = getattr(self._embedding_provider, "model_name", None)
        vectors: dict[str, list[float]] = {}
        for chunk in chunks:
            cached: list[float] | None = None
            if model_name:
                get_vector = getattr(self.store, "get_vector", None)
                if callable(get_vector):
                    cached = get_vector(chunk.content_hash, model_name)
            if cached is not None:
                vectors[chunk.id] = cached
                continue
            vec = self._embedding_provider.embed(chunk.content)
            if not isinstance(vec, list):
                continue
            vectors[chunk.id] = vec
            if model_name:
                set_vector = getattr(self.store, "set_vector", None)
                if callable(set_vector):
                    set_vector(chunk.content_hash, model_name, vec)
        return vectors

    def _build_link_plan(
        self,
        chunks: list[Chunk],
        vectors: dict[str, list[float]],
        existing_pairs: set[tuple[str, str]],
    ) -> list[StorageOperation]:
        by_node: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            by_node[c.node_path].append(c)

        paths = sorted(by_node.keys())
        operations: list[StorageOperation] = []

        for i, path_a in enumerate(paths):
            for path_b in paths[i + 1:]:
                if (path_a, path_b) in existing_pairs:
                    continue
                best_score = 0.0
                for chunk_a in by_node[path_a]:
                    vec_a = vectors.get(chunk_a.id)
                    if vec_a is None:
                        continue
                    for chunk_b in by_node[path_b]:
                        vec_b = vectors.get(chunk_b.id)
                        if vec_b is None:
                            continue
                        score = cosine_similarity(vec_a, vec_b)
                        if score > best_score:
                            best_score = score
                if best_score >= self._link_similarity_threshold:
                    operations.append(StorageOperation(
                        operation="create_link",
                        target_path=path_a,
                        links=[LinkInput(
                            target_path=path_b,
                            link_type="related",
                            reason=f"Silver similarity {best_score:.2f} ({LINK_DISCOVERY_SOURCE})",
                        )],
                        reasoning_summary=(
                            f"Cross-namespace Silver similarity {best_score:.2f} "
                            f"between {path_a} and {path_b}."
                        ),
                    ))
        return operations

    def _build_link_report(self, operations: list[StorageOperation]) -> str:
        if not operations:
            return "Link discovery: no cross-namespace Silver links found above threshold."
        lines = [f"Link discovery: {len(operations)} link(s) created."]
        for op in operations:
            for lnk in op.links:
                lines.append(f"  {op.target_path} → {lnk.target_path}  ({lnk.reason})")
        return "\n".join(lines)

    def _snapshot_branch(
        self, path: str
    ) -> tuple[list[Chunk], set[str], int | None]:
        """Read chunks, linked paths, and current node version in a single pass."""
        node = self.store.get_node(path)
        node_version = node.version if node is not None else None
        chunks = self.store.get_chunks_by_path(path, include_children=True)
        linked_paths = self._linked_paths_if_decay_enabled()
        return chunks, linked_paths, node_version

    # ── core planning (pure over the already-fetched chunk list) ──────────────

    def _build_plan(
        self,
        chunks: list[Chunk],
        path: str,
        *,
        linked_paths: set[str] | None = None,
    ) -> StorageOperationBatch:
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

        # Pass 3 — confidence decay for old unlinked Bronze/Silver chunks
        if self.decay_rate < 1.0 and self.decay_days > 0:
            _linked = linked_paths or set()
            for chunk in chunks:
                if chunk.status != "active":
                    continue
                if chunk.layer == "gold":
                    continue
                if chunk.id in stale_ids:
                    continue
                if self._is_decay_protected(chunk.node_path, _linked):
                    continue
                age_days = self._chunk_age_days(chunk)
                periods = int(age_days // self.decay_days)
                if periods == 0:
                    continue
                rate = chunk.decay_factor if chunk.decay_factor < 1.0 else self.decay_rate
                effective_confidence = chunk.confidence * (rate ** periods)
                if effective_confidence < self.stale_threshold:
                    operations.append(StorageOperation(
                        operation="mark_stale",
                        target_path=chunk.node_path,
                        chunk_ids=[chunk.id],
                        reasoning_summary=(
                            f"Confidence decayed below {self.stale_threshold:.2f} "
                            f"after {age_days:.0f} days without links."
                        ),
                    ))
                    stale_ids.add(chunk.id)

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
        decayed_by_node: dict[str, int] = defaultdict(int)

        for op in batch.operations:
            if op.operation == "mark_stale":
                if op.reasoning_summary and "decayed" in op.reasoning_summary:
                    decayed_by_node[op.target_path] += len(op.chunk_ids or [])
                else:
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
                decay = decayed_by_node[node_path]
                if dups or comp or decay:
                    lines.append(
                        f"  - {node_path}: "
                        f"{dups} exact duplicates marked stale, "
                        f"{comp} namespace compactions created, "
                        f"{decay} chunks decayed to stale"
                    )

        return "\n".join(lines)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _linked_paths_if_decay_enabled(self) -> set[str]:
        if self.decay_rate >= 1.0:
            return set()
        linked: set[str] = set()
        for lnk in self.store.list_links():
            linked.add(lnk.source_path)
            linked.add(lnk.target_path)
        return linked

    def _is_decay_protected(self, node_path: str, linked_paths: set[str]) -> bool:
        """Return True if node_path is directly linked or is a descendant of a linked path."""
        if node_path in linked_paths:
            return True
        for linked in linked_paths:
            if node_path.startswith(linked + "/"):
                return True
        return False

    def _chunk_age_days(self, chunk: Chunk) -> float:
        try:
            created = datetime.fromisoformat(chunk.created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - created).total_seconds() / 86400
        except (ValueError, TypeError):
            return 0.0

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
