from __future__ import annotations

from vertical_brain.core.context_lock import ContextLock
from vertical_brain.core.models import (
    ContextBudget,
    ContextPolicy,
    Link,
    LinkExpansionResult,
    NamespaceMap,
    SearchCandidateHandle,
    SearchContextResult,
    SearchResult,
)
from vertical_brain.core.namespace_map import NamespaceMapBuilder
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

    def expand_link(
        self,
        link_id: str,
        *,
        from_path: str | None = None,
        items_per_context: int = 6,
        include_ancestors: bool = True,
        link_expansion: str = "handles_only",
    ) -> LinkExpansionResult:
        link = _get_link(self.store, link_id)
        if link is None:
            raise ValueError(f"Link not found: {link_id}")

        expanded_path = _expanded_path(link, from_path)
        locked_context = self.lock.open_locked_context(
            expanded_path,
            policy=ContextPolicy(
                include_ancestors=include_ancestors,
                include_target=True,
                link_expansion=link_expansion,
            ),
            budget=ContextBudget(max_items=max(0, items_per_context)),
        )
        return LinkExpansionResult(
            link_id=link.id,
            source_path=from_path or link.source_path,
            expanded_path=expanded_path,
            link_type=link.link_type,
            reason=link.reason,
            locked_context=locked_context,
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


def _get_link(store, link_id: str) -> Link | None:
    get_link = getattr(store, "get_link", None)
    if callable(get_link):
        return get_link(link_id)
    return next((link for link in store.list_links() if link.id == link_id), None)


def _expanded_path(link: Link, from_path: str | None) -> str:
    if from_path is None:
        return link.target_path
    if from_path == link.source_path:
        return link.target_path
    if from_path == link.target_path:
        return link.source_path
    raise ValueError(f"Link {link.id} is not connected to {from_path}")
