from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vertical_brain.core.models import Chunk, Link, Node, SearchResult, utc_now
from vertical_brain.core.search import lexical_search


class JsonStore:
    def __init__(self, root: str | Path = "data"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.nodes_file = self.root / "nodes.json"
        self.chunks_file = self.root / "chunks.json"
        self.links_file = self.root / "links.json"
        self.namespace_roots_file = self.root / "namespaces" / "root.json"
        self.gold_dir = self.root / "gold"
        self.gold_dir.mkdir(parents=True, exist_ok=True)

        for file in [self.nodes_file, self.chunks_file, self.links_file]:
            if not file.exists():
                file.write_text("[]", encoding="utf-8")
        self._seed_namespace_roots()

    def _seed_namespace_roots(self) -> None:
        if not self.namespace_roots_file.exists():
            return

        payload = json.loads(self.namespace_roots_file.read_text(encoding="utf-8"))
        for root in payload.get("roots", []):
            if isinstance(root, str) and root:
                self.ensure_node(root)

    def _read(self, file: Path) -> list[dict[str, Any]]:
        return json.loads(file.read_text(encoding="utf-8"))

    def _write(self, file: Path, rows: list[dict[str, Any]]) -> None:
        file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def ensure_node(self, path: str) -> Node:
        nodes = self._read(self.nodes_file)
        existing = next((n for n in nodes if n["path"] == path), None)
        if existing:
            return Node(**self._clean_node_row(existing))

        existing_paths = {n["path"] for n in nodes}
        parts = path.split("/")
        target_row: dict[str, Any] | None = None

        for index in range(1, len(parts) + 1):
            node_path = "/".join(parts[:index])
            if node_path in existing_paths:
                if node_path == path:
                    target_row = next(n for n in nodes if n["path"] == node_path)
                continue

            parent_path = "/".join(parts[: index - 1]) or None
            node = Node(path=node_path, name=parts[index - 1], parent_path=parent_path)
            row = asdict(node)
            nodes.append(row)
            existing_paths.add(node_path)
            if node_path == path:
                target_row = row

        self._write(self.nodes_file, nodes)
        if target_row is None:
            target_row = next(n for n in nodes if n["path"] == path)
        return Node(**self._clean_node_row(target_row))

    def save_chunk(self, chunk: Chunk) -> Chunk:
        self.ensure_node(chunk.node_path)
        chunks = self._read(self.chunks_file)
        chunks.append(asdict(chunk))
        self._write(self.chunks_file, chunks)
        return chunk

    def save_link(self, link: Link) -> Link:
        self.ensure_node(link.source_path)
        self.ensure_node(link.target_path)
        links = self._read(self.links_file)
        existing = next(
            (
                row
                for row in links
                if row["source_path"] == link.source_path
                and row["target_path"] == link.target_path
                and row["link_type"] == link.link_type
            ),
            None,
        )
        if existing:
            return Link(**existing)
        links.append(asdict(link))
        self._write(self.links_file, links)
        return link

    def update_chunk(self, chunk: Chunk) -> Chunk:
        chunks = self._read(self.chunks_file)
        for index, row in enumerate(chunks):
            if row["id"] == chunk.id:
                chunks[index] = asdict(chunk)
                self._write(self.chunks_file, chunks)
                return chunk
        raise ValueError(f"Chunk not found: {chunk.id}")

    def gold_summary_path(self, path: str) -> Path:
        parts = path.split("/")
        return self.gold_dir.joinpath(*parts).with_suffix(".md")

    def get_node(self, path: str) -> Node | None:
        existing = next((row for row in self._read(self.nodes_file) if row["path"] == path), None)
        if existing is None:
            return None
        return Node(**self._clean_node_row(existing))

    def list_nodes(self) -> list[Node]:
        return [Node(**self._clean_node_row(row)) for row in self._read(self.nodes_file)]

    def _clean_node_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in row.items() if k in {"id", "path", "name", "parent_path", "node_type", "created_at", "updated_at"}}

    def list_chunks(self) -> list[Chunk]:
        return [Chunk(**row) for row in self._read(self.chunks_file)]

    def list_links(self) -> list[Link]:
        return [Link(**row) for row in self._read(self.links_file)]

    def get_link(self, link_id: str) -> Link | None:
        existing = next((row for row in self._read(self.links_file) if row["id"] == link_id), None)
        if existing is None:
            return None
        return Link(**existing)

    def get_peer_links(self, path: str) -> list[Link]:
        return [
            link
            for link in self.list_links()
            if link.link_type == "peer" and (link.source_path == path or link.target_path == path)
        ]

    def get_peer_paths(self, path: str) -> list[str]:
        peer_paths: list[str] = []
        for link in self.get_peer_links(path):
            peer_path = link.target_path if link.source_path == path else link.source_path
            if peer_path not in peer_paths:
                peer_paths.append(peer_path)
        return peer_paths

    def get_chunks_by_path(self, path: str, include_children: bool = False) -> list[Chunk]:
        chunks = self.list_chunks()
        if include_children:
            return [c for c in chunks if c.node_path == path or c.node_path.startswith(path + "/")]
        return [c for c in chunks if c.node_path == path]

    def search(
        self,
        query: str,
        *,
        root_path: str | None = None,
        limit: int = 10,
        include_stale: bool = False,
    ) -> list[SearchResult]:
        return lexical_search(
            chunks=self.list_chunks(),
            query=query,
            root_path=root_path,
            limit=limit,
            include_stale=include_stale,
        )

    def get_ancestors(self, path: str) -> list[str]:
        parts = path.split("/")
        ancestors = []
        for i in range(1, len(parts)):
            ancestors.append("/".join(parts[:i]))
        return ancestors

    def tree_text(self) -> str:
        paths = sorted(n.path for n in self.list_nodes())
        if not paths:
            return "(empty tree)"

        lines = []
        for path in paths:
            depth = path.count("/")
            lines.append("  " * depth + path.split("/")[-1])
        return "\n".join(lines)
