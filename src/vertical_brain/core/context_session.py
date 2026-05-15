from __future__ import annotations

from vertical_brain.core.context_lock import ContextLock
from vertical_brain.core.models import (
    ContextBudget,
    ContextPolicy,
    SearchCandidateHandle,
    SearchContextResult,
    SearchResult,
)
from vertical_brain.core.search import BrainSearch


class ContextSession:
    """Model-facing read session.

    Search is used only to find candidate namespace paths. The model-facing
    content still comes from locked vertical context capsules.
    """

    def __init__(self, store):
        self.store = store
        self.search = BrainSearch(store)
        self.lock = ContextLock(store)

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
