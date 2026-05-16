from __future__ import annotations

from typing import TYPE_CHECKING

from vertical_brain.core.gold import parse_gold_content
from vertical_brain.core.models import Chunk, Link, LinkHandle, NamespaceMap, NamespaceMapNode, Node

if TYPE_CHECKING:
    from vertical_brain.storage.protocol import StorageProvider


class NamespaceMapBuilder:
    """Builds a model-facing map without exposing raw chunk content."""

    def __init__(self, store: "StorageProvider"):
        self.store = store

    def build(
        self,
        *,
        root_path: str | None = None,
        max_depth: int | None = None,
        summary_max_chars: int = 240,
    ) -> NamespaceMap:
        nodes = [node for node in self.store.list_nodes() if _path_in_scope(node.path, root_path)]
        visible_nodes, omitted_nodes = _apply_depth_limit(nodes, root_path=root_path, max_depth=max_depth)
        visible_paths = {node.path for node in visible_nodes}
        children_by_parent = _children_by_parent(visible_nodes)
        chunks = self.store.list_chunks()
        links = self.store.list_links()
        chunks_by_path = _chunks_by_path(chunks)
        subtree_chunk_count_by_path = _subtree_chunk_counts(chunks)
        links_by_path = _links_by_path(links)

        map_nodes: list[NamespaceMapNode] = []
        gold_chunks_by_path: dict[str, list] = {}
        for chunk in chunks:
            if chunk.layer == "gold" and chunk.status == "active":
                gold_chunks_by_path.setdefault(chunk.node_path, []).append(chunk)

        for node in sorted(visible_nodes, key=lambda item: item.path):
            node_gold = sorted(gold_chunks_by_path.get(node.path, []), key=lambda c: c.created_at)
            gold_aspects: list[str] = []
            for chunk in node_gold:
                gold_aspects.extend(parse_gold_content(chunk.content))
            gold_text = " | ".join(gold_aspects)
            gold_summary, omitted_summary_chars = _truncate(gold_text, summary_max_chars)
            node_chunks = chunks_by_path.get(node.path, [])
            node_links = links_by_path.get(node.path, [])
            map_nodes.append(
                NamespaceMapNode(
                    path=node.path,
                    name=node.name,
                    parent_path=node.parent_path if node.parent_path in visible_paths else None,
                    depth=_relative_depth(node.path, root_path),
                    children=children_by_parent.get(node.path, []),
                    chunk_count=len(node_chunks),
                    active_chunk_count=sum(1 for chunk in node_chunks if chunk.status == "active"),
                    stale_chunk_count=sum(1 for chunk in node_chunks if chunk.status != "active"),
                    subtree_chunk_count=subtree_chunk_count_by_path.get(node.path, 0),
                    link_count=len(node_links),
                    link_handles=_link_handles_for_path(node_links, node.path),
                    gold_summary=gold_summary,
                    omitted_summary_chars=omitted_summary_chars,
                    updated_at=node.updated_at,
                )
            )

        return NamespaceMap(
            root_path=root_path,
            nodes=map_nodes,
            omitted_nodes=omitted_nodes,
            summary_max_chars=summary_max_chars,
        )


def _path_in_scope(path: str, root_path: str | None) -> bool:
    if root_path is None or not root_path:
        return True
    return path == root_path or path.startswith(root_path + "/")


def _apply_depth_limit(
    nodes: list[Node],
    *,
    root_path: str | None,
    max_depth: int | None,
) -> tuple[list[Node], int]:
    if max_depth is None:
        return nodes, 0

    visible_nodes = [
        node
        for node in nodes
        if _relative_depth(node.path, root_path) <= max_depth
    ]
    return visible_nodes, len(nodes) - len(visible_nodes)


def _relative_depth(path: str, root_path: str | None) -> int:
    if root_path is None or not root_path:
        return path.count("/")
    if path == root_path:
        return 0
    return path[len(root_path) + 1 :].count("/") + 1


def _children_by_parent(nodes: list[Node]) -> dict[str, list[str]]:
    visible_paths = {node.path for node in nodes}
    children: dict[str, list[str]] = {}
    for node in nodes:
        if node.parent_path in visible_paths:
            children.setdefault(node.parent_path, []).append(node.path)
    return {path: sorted(paths) for path, paths in children.items()}


def _chunks_by_path(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    by_path: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_path.setdefault(chunk.node_path, []).append(chunk)
    return by_path


def _subtree_chunk_counts(chunks: list[Chunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        parts = chunk.node_path.split("/")
        for index in range(1, len(parts) + 1):
            ancestor = "/".join(parts[:index])
            counts[ancestor] = counts.get(ancestor, 0) + 1
    return counts


def _links_for_path(links: list[Link], path: str) -> list[Link]:
    return [link for link in links if link.source_path == path or link.target_path == path]


def _links_by_path(links: list[Link]) -> dict[str, list[Link]]:
    by_path: dict[str, list[Link]] = {}
    for link in links:
        by_path.setdefault(link.source_path, []).append(link)
        if link.target_path != link.source_path:
            by_path.setdefault(link.target_path, []).append(link)
    return by_path


def _link_handles_for_path(links: list[Link], path: str) -> list[LinkHandle]:
    handles: list[LinkHandle] = []
    for link in _links_for_path(links, path):
        peer_path = link.target_path if link.source_path == path else link.source_path
        handles.append(
            LinkHandle(
                link_id=link.id,
                target_path=peer_path,
                link_type=link.link_type,
                reason=link.reason,
            )
        )
    return handles


def _truncate(text: str, max_chars: int) -> tuple[str, int]:
    if max_chars <= 0:
        return "", len(text)
    if len(text) <= max_chars:
        return text, 0
    if max_chars <= 3:
        return text[:max_chars], len(text) - max_chars
    return text[: max_chars - 3].rstrip() + "...", len(text) - max_chars
