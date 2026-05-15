from __future__ import annotations

from vertical_brain.core.models import (
    ContextBudget,
    ContextItem,
    ContextPolicy,
    LinkHandle,
    LockedContext,
)
from vertical_brain.storage.json_store import JsonStore


class ContextLock:
    """Builds a bounded vertical context capsule for a model.

    Horizontal links are handles by default. Their content is not expanded into
    the prompt unless the caller explicitly asks for expanded links.
    """

    def __init__(self, store: JsonStore):
        self.store = store

    def open_locked_context(
        self,
        target_path: str,
        policy: ContextPolicy | None = None,
        budget: ContextBudget | None = None,
    ) -> LockedContext:
        policy = policy or ContextPolicy()
        budget = budget or ContextBudget()
        items: list[ContextItem] = []
        omitted_items = 0

        if policy.include_ancestors:
            for ancestor_path in self.store.get_ancestors(target_path):
                ancestor = self.store.get_node(ancestor_path)
                if ancestor and ancestor.gold_aspects:
                    omitted_items += self._append_with_budget(
                        items,
                        ContextItem(
                            path=ancestor.path,
                            layer="gold",
                            content=" | ".join(ancestor.gold_aspects),
                            source="node",
                        ),
                        budget,
                    )

        target = self.store.get_node(target_path)
        if policy.include_target and target and target.gold_aspects:
            omitted_items += self._append_with_budget(
                items,
                ContextItem(
                    path=target.path,
                    layer="gold",
                    content=" | ".join(target.gold_aspects),
                    source="node",
                ),
                budget,
            )

        if policy.include_target:
            for chunk in self.store.get_chunks_by_path(target_path):
                if chunk.status == "active":
                    omitted_items += self._append_with_budget(
                        items,
                        ContextItem(
                            path=chunk.node_path,
                            layer=chunk.layer,
                            content=chunk.content,
                            source="chunk",
                        ),
                        budget,
                    )

        link_handles = self._link_handles(target_path) if policy.link_expansion != "none" else []
        if policy.link_expansion == "expanded":
            for handle in link_handles:
                for chunk in self.store.get_chunks_by_path(handle.target_path):
                    if chunk.status == "active":
                        omitted_items += self._append_with_budget(
                            items,
                            ContextItem(
                                path=chunk.node_path,
                                layer=f"linked:{chunk.layer}",
                                content=chunk.content,
                                source="linked_chunk",
                            ),
                            budget,
                        )

        return LockedContext(
            target_path=target_path,
            items=items,
            link_handles=link_handles,
            budget=budget,
            policy=policy,
            omitted_items=omitted_items,
        )

    def build_context(
        self,
        target_path: str,
        include_ancestors: bool = True,
        include_peer_links: bool = True,
    ) -> list[str]:
        link_expansion = "expanded" if include_peer_links else "none"
        locked_context = self.open_locked_context(
            target_path,
            policy=ContextPolicy(
                include_ancestors=include_ancestors,
                include_target=True,
                link_expansion=link_expansion,
            ),
        )
        return locked_context.as_prompt_lines()

    def _append_with_budget(
        self,
        items: list[ContextItem],
        item: ContextItem,
        budget: ContextBudget,
    ) -> int:
        if len(items) >= budget.max_items:
            return 1
        items.append(item)
        return 0

    def _link_handles(self, target_path: str) -> list[LinkHandle]:
        handles: list[LinkHandle] = []
        for link in self.store.get_peer_links(target_path):
            peer_path = link.target_path if link.source_path == target_path else link.source_path
            handles.append(
                LinkHandle(
                    link_id=link.id,
                    target_path=peer_path,
                    link_type=link.link_type,
                    reason=link.reason,
                )
            )
        return handles
