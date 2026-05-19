"""Backend-agnostic storage derivations.

Every method here is a pure function of other StorageProvider primitives —
no backend-specific I/O. Both SQLiteStore and JsonStore mix this in, so the
logic lives in exactly one place and the two backends cannot drift apart.

Concrete stores must still provide the primitives these build on:
``get_peer_links`` and ``list_nodes``.
"""
from __future__ import annotations


class StorageDerivationsMixin:
    """Shared read-only derivations layered on top of backend primitives."""

    def get_peer_paths(self, path: str) -> list[str]:
        peer_paths: list[str] = []
        for link in self.get_peer_links(path):  # type: ignore[attr-defined]
            peer_path = link.target_path if link.source_path == path else link.source_path
            if peer_path not in peer_paths:
                peer_paths.append(peer_path)
        return peer_paths

    def get_ancestors(self, path: str) -> list[str]:
        parts = path.split("/")
        return ["/".join(parts[:i]) for i in range(1, len(parts))]

    def tree_text(self) -> str:
        paths = sorted(n.path for n in self.list_nodes())  # type: ignore[attr-defined]
        if not paths:
            return "(empty tree)"
        lines = []
        for path in paths:
            depth = path.count("/")
            lines.append("  " * depth + path.split("/")[-1])
        return "\n".join(lines)
