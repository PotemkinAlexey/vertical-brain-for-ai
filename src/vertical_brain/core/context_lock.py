from __future__ import annotations

from vertical_brain.storage.json_store import JsonStore


class ContextLock:
    """Builds allowed context for a query.

    MVP behavior:
    - include ancestor Gold summaries
    - include target path chunks
    - include explicitly linked peer node chunks
    - exclude all other branches
    """

    def __init__(self, store: JsonStore):
        self.store = store

    def build_context(self, target_path: str) -> list[str]:
        context: list[str] = []

        for ancestor_path in self.store.get_ancestors(target_path):
            ancestor = self.store.get_node(ancestor_path)
            if ancestor and ancestor.gold_summary:
                context.append(f"[{ancestor.path}][gold_summary] {ancestor.gold_summary}")

        target = self.store.get_node(target_path)
        if target and target.gold_summary:
            context.append(f"[{target.path}][gold_summary] {target.gold_summary}")

        self._append_active_chunks(context, target_path)

        for peer_path in self.store.get_peer_paths(target_path):
            self._append_active_chunks(context, peer_path, label_prefix="peer:")

        return context

    def _append_active_chunks(
        self,
        context: list[str],
        path: str,
        label_prefix: str = "",
    ) -> None:
        for chunk in self.store.get_chunks_by_path(path):
            if chunk.status == "active":
                context.append(f"[{chunk.node_path}][{label_prefix}{chunk.layer}] {chunk.content}")
