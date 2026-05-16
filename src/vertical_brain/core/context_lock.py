from __future__ import annotations

from typing import TYPE_CHECKING

from vertical_brain.core.gold import parse_gold_content
from vertical_brain.core.models import (
    ContextBudget,
    ContextItem,
    ContextPolicy,
    LinkHandle,
    LockedContext,
)

if TYPE_CHECKING:
    from vertical_brain.storage.protocol import StorageProvider


_SYMMETRIC_LINK_TYPES = {"peer", "related_to"}
_LAYER_PRIORITY: dict[str, int] = {"gold": 0, "silver": 1, "bronze": 2}


class ContextLock:
    """Builds a bounded vertical context capsule for a model.

    Horizontal links are handles by default. Their content is not expanded into
    the prompt unless the caller explicitly asks for expanded links.
    """

    def __init__(self, store: "StorageProvider"):
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
                gold_chunks = sorted(
                    [c for c in self.store.get_chunks_by_path(ancestor_path) if c.layer == "gold" and c.status == "active"],
                    key=lambda c: c.created_at,
                )
                if gold_chunks:
                    omitted_items += self._append_with_budget(
                        items,
                        ContextItem(
                            path=ancestor_path,
                            layer="gold",
                            content=self._render_gold(gold_chunks),
                            source="chunk",
                        ),
                        budget,
                    )

        target_gold = sorted(
            [c for c in self.store.get_chunks_by_path(target_path) if c.layer == "gold" and c.status == "active"],
            key=lambda c: c.created_at,
        )
        if policy.include_target and target_gold:
            omitted_items += self._append_with_budget(
                items,
                ContextItem(
                    path=target_path,
                    layer="gold",
                    content=self._render_gold(target_gold),
                    source="chunk",
                ),
                budget,
            )

        if policy.include_target:
            for chunk in sorted(
                self.store.get_chunks_by_path(target_path),
                key=lambda c: (_LAYER_PRIORITY.get(c.layer, 3), c.created_at),
            ):
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
            visited: set[str] = {target_path}
            for handle in link_handles:
                if handle.target_path in visited:
                    continue
                visited.add(handle.target_path)
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

    def _render_gold(self, gold_chunks: list) -> str:
        aspects: list[str] = []
        for chunk in gold_chunks:
            aspects.extend(parse_gold_content(chunk.content))
        return " | ".join(aspects)

    def expand_link(self, link_id: str, from_path: str | None = None) -> str:
        """Resolve the far endpoint of a link.

        For directional (non-symmetric) link types, ``from_path`` is required
        to disambiguate which endpoint the caller is expanding from.
        """
        link = self.store.get_link(link_id)
        if link is None:
            raise ValueError(f"Link not found: {link_id}")

        if link.link_type in _SYMMETRIC_LINK_TYPES:
            if from_path is None:
                return link.target_path
            if from_path == link.source_path:
                return link.target_path
            if from_path == link.target_path:
                return link.source_path
            raise ValueError(f"from_path {from_path} is not an endpoint of link {link_id}")

        if from_path is None:
            raise ValueError(
                f"from_path is required to expand this link because link type "
                f"'{link.link_type}' is directional."
            )
        if from_path == link.source_path:
            return link.target_path
        if from_path == link.target_path:
            return link.source_path
        raise ValueError(f"from_path {from_path} is not an endpoint of link {link_id}")

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
