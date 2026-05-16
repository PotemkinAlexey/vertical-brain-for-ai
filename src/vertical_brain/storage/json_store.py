from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator
from uuid import uuid4

from vertical_brain.core.models import Chunk, Link, Node, SearchResult, utc_now
from vertical_brain.core.search import lexical_search

if TYPE_CHECKING:
    from vertical_brain.core.models import OperationResult, StorageOperation


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
        self.audit_file = self.root / "operation_audit.jsonl"
        self._transaction_depth = 0

        for file in [self.nodes_file, self.chunks_file, self.links_file]:
            if not file.exists():
                self._write_json(file, [])
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
        self._write_json(file, rows, indent=2)

    def _write_json(self, file: Path, payload: Any, *, indent: int | None = None) -> None:
        """Atomically replace a JSON file.

        JsonStore is intended for local/dev use, but atomic replace avoids
        corrupting files when a process is interrupted during a write.
        """
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_name(f".{file.name}.{uuid4().hex}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")
        tmp.replace(file)

    def _transaction_files(self) -> list[Path]:
        return [
            self.nodes_file,
            self.chunks_file,
            self.links_file,
            self.audit_file,
            self.root / "embedding_schema.json",
            self.root / "vector_cache.json",
        ]

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Rollback file-backed state if a batch fails mid-transaction.

        This is not a substitute for SQLite's crash-safe ACID behavior, but it
        preserves the executor's all-or-nothing guarantee for normal in-process
        exceptions and keeps the JSON backend useful for deterministic tests.
        """
        outermost = self._transaction_depth == 0
        snapshot: dict[Path, bytes | None] = {}
        if outermost:
            for file in self._transaction_files():
                snapshot[file] = file.read_bytes() if file.exists() else None

        self._transaction_depth += 1
        try:
            yield
        except Exception:
            self._transaction_depth -= 1
            if outermost:
                for file, content in snapshot.items():
                    if content is None:
                        try:
                            file.unlink()
                        except FileNotFoundError:
                            pass
                    else:
                        file.parent.mkdir(parents=True, exist_ok=True)
                        file.write_bytes(content)
            raise
        else:
            self._transaction_depth -= 1

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
        known = {"id", "path", "name", "parent_path", "node_type", "is_dirty", "version", "created_at", "updated_at"}
        data = {k: v for k, v in row.items() if k in known}
        data.setdefault("is_dirty", False)
        data.setdefault("version", 0)
        return data

    def update_node(self, node: Node) -> Node:
        nodes = self._read(self.nodes_file)
        for index, row in enumerate(nodes):
            if row["path"] == node.path:
                from dataclasses import asdict as _asdict
                nodes[index] = _asdict(node)
                self._write(self.nodes_file, nodes)
                return node
        raise ValueError(f"Node not found: {node.path}")

    def list_chunks(self) -> list[Chunk]:
        return [self._chunk_from_row(row) for row in self._read(self.chunks_file)]

    def _chunk_from_row(self, row: dict[str, Any]) -> Chunk:
        known = {
            "node_path", "content", "layer", "content_type", "status", "source",
            "confidence", "lineage", "id", "created_at", "updated_at",
            "chunk_key", "content_hash", "supersedes", "valid_from", "valid_to",
            "decay_factor",
        }
        data = {k: v for k, v in row.items() if k in known}
        data.setdefault("chunk_key", None)
        data.setdefault("content_hash", "")
        data.setdefault("supersedes", [])
        data.setdefault("valid_from", row.get("created_at") or "")
        data.setdefault("valid_to", None)
        data.setdefault("decay_factor", 1.0)
        return Chunk(**data)

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

    def log_audit(self, operation: "StorageOperation", result: "OperationResult") -> None:
        record = {
            "id": str(uuid4()),
            "operation_type": operation.operation,
            "target_path": operation.target_path,
            "payload_json": operation.to_json(),
            "result_json": result.to_json(),
            "status": result.status,
            "reasoning_summary": operation.reasoning_summary,
            "created_at": utc_now(),
        }
        with self.audit_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def list_audit(self) -> list[dict[str, Any]]:
        if not self.audit_file.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.audit_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def get_embedding_schema(self) -> dict | None:
        schema_file = self.root / "embedding_schema.json"
        if not schema_file.exists():
            return None
        return json.loads(schema_file.read_text(encoding="utf-8"))

    def set_embedding_schema(self, model_name: str, vector_dimension: int) -> None:
        schema_file = self.root / "embedding_schema.json"
        self._write_json(
            schema_file,
            {"model_name": model_name, "vector_dimension": vector_dimension},
        )

    def _read_vector_cache(self) -> dict[str, Any]:
        cache_file = self.root / "vector_cache.json"
        if not cache_file.exists():
            return {}
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_vector_cache(self, cache: dict[str, Any]) -> None:
        self._write_json(self.root / "vector_cache.json", cache)

    def get_vector(self, content_hash: str, model_name: str) -> list[float] | None:
        entry = self._read_vector_cache().get(content_hash, {})
        return entry.get(model_name)

    def set_vector(self, content_hash: str, model_name: str, vector: list[float]) -> None:
        cache = self._read_vector_cache()
        cache.setdefault(content_hash, {})[model_name] = vector
        self._write_vector_cache(cache)

    def delete_vectors_for_model(self, model_name: str) -> int:
        cache = self._read_vector_cache()
        count = sum(1 for entry in cache.values() if model_name in entry)
        for entry in cache.values():
            entry.pop(model_name, None)
        self._write_vector_cache(cache)
        return count

    def rename_namespace(self, old_prefix: str, new_prefix: str) -> None:
        chunks = self._read(self.chunks_file)
        for row in chunks:
            if row["node_path"] == old_prefix or row["node_path"].startswith(old_prefix + "/"):
                row["node_path"] = new_prefix + row["node_path"][len(old_prefix):]
        self._write(self.chunks_file, chunks)

        links = self._read(self.links_file)
        for row in links:
            if row["source_path"] == old_prefix or row["source_path"].startswith(old_prefix + "/"):
                row["source_path"] = new_prefix + row["source_path"][len(old_prefix):]
            if row["target_path"] == old_prefix or row["target_path"].startswith(old_prefix + "/"):
                row["target_path"] = new_prefix + row["target_path"][len(old_prefix):]
        self._write(self.links_file, links)

        nodes = self._read(self.nodes_file)
        for row in nodes:
            if row["path"] == old_prefix or row["path"].startswith(old_prefix + "/"):
                row["path"] = new_prefix + row["path"][len(old_prefix):]
                parts = row["path"].split("/")
                row["name"] = parts[-1]
                row["parent_path"] = "/".join(parts[:-1]) if len(parts) > 1 else None
            elif row.get("parent_path") and (
                row["parent_path"] == old_prefix
                or row["parent_path"].startswith(old_prefix + "/")
            ):
                row["parent_path"] = new_prefix + row["parent_path"][len(old_prefix):]
        self._write(self.nodes_file, nodes)

    def tree_text(self) -> str:
        paths = sorted(n.path for n in self.list_nodes())
        if not paths:
            return "(empty tree)"

        lines = []
        for path in paths:
            depth = path.count("/")
            lines.append("  " * depth + path.split("/")[-1])
        return "\n".join(lines)
