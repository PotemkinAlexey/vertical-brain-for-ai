from __future__ import annotations

from datetime import date

from vertical_brain.core.context_lock import ContextLock
from vertical_brain.core.models import (
    ContextBudget,
    ContextPolicy,
    NamespaceMap,
    SearchCandidateHandle,
    SearchContextResult,
    SearchResult,
)
from vertical_brain.core.namespace_map import NamespaceMapBuilder
from vertical_brain.core.search import BrainSearch
from vertical_brain.llm.embedding import EmbeddingProvider


class ContextSession:
    """Model-facing read session.

    Search is used only to find candidate namespace paths. The model-facing
    content still comes from locked vertical context capsules.
    """

    def __init__(self, store):
        self.store = store
        self.search = BrainSearch(store)
        self.lock = ContextLock(store)
        self.map_builder = NamespaceMapBuilder(store)

    def namespace_map(
        self,
        *,
        root_path: str | None = None,
        max_depth: int | None = None,
        summary_max_chars: int = 240,
    ) -> NamespaceMap:
        return self.map_builder.build(
            root_path=root_path,
            max_depth=max_depth,
            summary_max_chars=summary_max_chars,
        )

    def session_prompt(
        self,
        *,
        root_path: str | None = None,
        max_depth: int | None = None,
        summary_max_chars: int = 200,
    ) -> str:
        """Return a compact LLM-ready orientation text for the start of a session."""
        nm = self.namespace_map(
            root_path=root_path,
            max_depth=max_depth,
            summary_max_chars=summary_max_chars,
        )
        total_active = sum(n.active_chunk_count for n in nm.nodes)
        unique_link_ids: set[str] = set()
        for n in nm.nodes:
            for h in n.link_handles:
                unique_link_ids.add(h.link_id)
        total_links = len(unique_link_ids)

        lines: list[str] = [
            f"VERTICAL BRAIN — {date.today().isoformat()}",
            f"{len(nm.nodes)} nodes · {total_active} active chunks · {total_links} links",
            "",
        ]
        for node in nm.nodes:
            indent = "  " * node.depth
            head = f"{indent}[{node.path}]"
            parts: list[str] = []
            if node.gold_summary:
                parts.append(node.gold_summary)
            stats: list[str] = []
            if node.active_chunk_count:
                stats.append(f"{node.active_chunk_count} active")
            if node.stale_chunk_count:
                stats.append(f"{node.stale_chunk_count} stale")
            peer_links = [h for h in node.link_handles if h.link_type == "peer"]
            if peer_links:
                targets = ", ".join(f"→{h.target_path}" for h in peer_links)
                stats.append(targets)
            if stats:
                parts.append("— " + ", ".join(stats))
            suffix = "  ".join(parts) if parts else "(empty)"
            lines.append(f"{head} {suffix}")

        if nm.omitted_nodes:
            lines.append(f"\n({nm.omitted_nodes} nodes omitted due to depth limit)")
        return "\n".join(lines)

    def search_locked_context(
        self,
        query: str,
        *,
        root_path: str | None = None,
        search_limit: int = 10,
        context_limit: int = 3,
        items_per_context: int = 6,
        include_ancestors: bool = True,
        link_expansion: str = "handles_only",
    ) -> SearchContextResult:
        results = self.search.search(
            query,
            root_path=root_path,
            limit=search_limit,
            include_stale=False,
        )
        handles = [_handle_from_result(result) for result in results]
        candidate_paths = _unique_paths(results)
        selected_paths = candidate_paths[: max(0, context_limit)]
        policy = ContextPolicy(
            include_ancestors=include_ancestors,
            include_target=True,
            link_expansion=link_expansion,
        )
        budget = ContextBudget(max_items=max(0, items_per_context))
        locked_contexts = [
            self.lock.open_locked_context(path, policy=policy, budget=budget)
            for path in selected_paths
        ]

        return SearchContextResult(
            query=query,
            candidate_handles=handles,
            locked_contexts=locked_contexts,
            omitted_candidates=max(0, len(candidate_paths) - len(selected_paths)),
        )


    def search_locked_context_semantic(
        self,
        query: str,
        provider: EmbeddingProvider,
        *,
        root_path: str | None = None,
        search_limit: int = 10,
        context_limit: int = 3,
        items_per_context: int = 6,
        include_ancestors: bool = True,
        link_expansion: str = "handles_only",
        threshold: float = 0.0,
    ) -> SearchContextResult:
        from vertical_brain.core.embedding_search import EmbeddingSearch
        results = EmbeddingSearch(self.store, provider).search(
            query,
            root_path=root_path,
            limit=search_limit,
            threshold=threshold,
        )
        handles = [_handle_from_result(result) for result in results]
        candidate_paths = _unique_paths(results)
        selected_paths = candidate_paths[: max(0, context_limit)]
        policy = ContextPolicy(
            include_ancestors=include_ancestors,
            include_target=True,
            link_expansion=link_expansion,
        )
        budget = ContextBudget(max_items=max(0, items_per_context))
        locked_contexts = [
            self.lock.open_locked_context(path, policy=policy, budget=budget)
            for path in selected_paths
        ]
        return SearchContextResult(
            query=query,
            candidate_handles=handles,
            locked_contexts=locked_contexts,
            omitted_candidates=max(0, len(candidate_paths) - len(selected_paths)),
        )


def _handle_from_result(result: SearchResult) -> SearchCandidateHandle:
    return SearchCandidateHandle(
        path=result.path,
        source=result.source,
        score=result.score,
        chunk_id=result.chunk_id,
        layer=result.layer,
        content_type=result.content_type,
        status=result.status,
    )


def _unique_paths(results: list[SearchResult]) -> list[str]:
    paths: list[str] = []
    for result in results:
        if result.path not in paths:
            paths.append(result.path)
    return paths
