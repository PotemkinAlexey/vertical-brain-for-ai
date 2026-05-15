from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from vertical_brain.core.models import Chunk, Link, Node, utc_now


class SQLiteStore:
    def __init__(self, root: str | Path = "data", db_name: str = "vertical_brain.sqlite"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_file = self.root / db_name
        self.namespace_roots_file = self.root / "namespaces" / "root.json"
        self.gold_dir = self.root / "gold"
        self.gold_dir.mkdir(parents=True, exist_ok=True)
        self._transaction_depth = 0

        self.conn = sqlite3.connect(self.db_file)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        self._seed_namespace_roots()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                parent_path TEXT,
                node_type TEXT NOT NULL,
                gold_summary TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                node_path TEXT NOT NULL,
                content TEXT NOT NULL,
                layer TEXT NOT NULL,
                content_type TEXT NOT NULL,
                status TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                lineage_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS links (
                id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                target_path TEXT NOT NULL,
                link_type TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(source_path, target_path, link_type)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_node_path ON chunks(node_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status);
            CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_path);
            CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_path);
            """
        )
        self.conn.commit()

    def _seed_namespace_roots(self) -> None:
        if not self.namespace_roots_file.exists():
            return

        payload = json.loads(self.namespace_roots_file.read_text(encoding="utf-8"))
        for root in payload.get("roots", []):
            if isinstance(root, str) and root:
                self.ensure_node(root)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        outermost = self._transaction_depth == 0
        if outermost:
            self.conn.execute("BEGIN")
        self._transaction_depth += 1
        try:
            yield
        except Exception:
            self._transaction_depth -= 1
            if outermost:
                self.conn.rollback()
            raise
        else:
            self._transaction_depth -= 1
            if outermost:
                self.conn.commit()

    def _commit_if_needed(self) -> None:
        if self._transaction_depth == 0:
            self.conn.commit()

    def ensure_node(self, path: str) -> Node:
        existing = self.get_node(path)
        if existing is not None:
            return existing

        parts = path.split("/")
        target: Node | None = None
        for index in range(1, len(parts) + 1):
            node_path = "/".join(parts[:index])
            existing = self.get_node(node_path)
            if existing is not None:
                if node_path == path:
                    target = existing
                continue

            parent_path = "/".join(parts[: index - 1]) or None
            node = Node(path=node_path, name=parts[index - 1], parent_path=parent_path)
            self.conn.execute(
                """
                INSERT INTO nodes (
                    id, path, name, parent_path, node_type, gold_summary, created_at, updated_at
                )
                VALUES (
                    :id, :path, :name, :parent_path, :node_type, :gold_summary, :created_at, :updated_at
                )
                """,
                asdict(node),
            )
            target = node if node_path == path else target

        self._commit_if_needed()
        if target is None:
            target = self.get_node(path)
        if target is None:
            raise ValueError(f"Node not found after ensure_node: {path}")
        return target

    def save_chunk(self, chunk: Chunk) -> Chunk:
        self.ensure_node(chunk.node_path)
        row = asdict(chunk)
        row["lineage_json"] = json.dumps(chunk.lineage, ensure_ascii=False)
        del row["lineage"]
        self.conn.execute(
            """
            INSERT INTO chunks (
                id, node_path, content, layer, content_type, status, source,
                confidence, lineage_json, created_at, updated_at
            )
            VALUES (
                :id, :node_path, :content, :layer, :content_type, :status, :source,
                :confidence, :lineage_json, :created_at, :updated_at
            )
            """,
            row,
        )
        self._commit_if_needed()
        return chunk

    def save_link(self, link: Link) -> Link:
        self.ensure_node(link.source_path)
        self.ensure_node(link.target_path)
        existing = self._get_link_by_relationship(link.source_path, link.target_path, link.link_type)
        if existing is not None:
            return existing

        try:
            self.conn.execute(
                """
                INSERT INTO links (id, source_path, target_path, link_type, reason, created_at)
                VALUES (:id, :source_path, :target_path, :link_type, :reason, :created_at)
                """,
                asdict(link),
            )
        except sqlite3.IntegrityError:
            existing = self._get_link_by_relationship(link.source_path, link.target_path, link.link_type)
            if existing is not None:
                return existing
            raise
        self._commit_if_needed()
        return link

    def update_chunk(self, chunk: Chunk) -> Chunk:
        row = asdict(chunk)
        row["lineage_json"] = json.dumps(chunk.lineage, ensure_ascii=False)
        del row["lineage"]
        cursor = self.conn.execute(
            """
            UPDATE chunks
            SET node_path = :node_path,
                content = :content,
                layer = :layer,
                content_type = :content_type,
                status = :status,
                source = :source,
                confidence = :confidence,
                lineage_json = :lineage_json,
                created_at = :created_at,
                updated_at = :updated_at
            WHERE id = :id
            """,
            row,
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Chunk not found: {chunk.id}")
        self._commit_if_needed()
        return chunk

    def update_node_gold_summary(self, path: str, summary: str) -> Node:
        self.ensure_node(path)
        updated_at = utc_now()
        cursor = self.conn.execute(
            """
            UPDATE nodes
            SET gold_summary = ?, updated_at = ?
            WHERE path = ?
            """,
            (summary, updated_at, path),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Node not found: {path}")
        self._commit_if_needed()
        self._write_gold_summary_markdown(path, summary)
        node = self.get_node(path)
        if node is None:
            raise ValueError(f"Node not found: {path}")
        return node

    def gold_summary_path(self, path: str) -> Path:
        parts = path.split("/")
        return self.gold_dir.joinpath(*parts).with_suffix(".md")

    def _write_gold_summary_markdown(self, path: str, summary: str) -> None:
        file = self.gold_summary_path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(f"# {path}\n\n{summary}\n", encoding="utf-8")

    def get_node(self, path: str) -> Node | None:
        row = self.conn.execute("SELECT * FROM nodes WHERE path = ?", (path,)).fetchone()
        if row is None:
            return None
        return Node(**dict(row))

    def list_nodes(self) -> list[Node]:
        rows = self.conn.execute("SELECT * FROM nodes ORDER BY rowid").fetchall()
        return [Node(**dict(row)) for row in rows]

    def list_chunks(self) -> list[Chunk]:
        rows = self.conn.execute("SELECT * FROM chunks ORDER BY rowid").fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def list_links(self) -> list[Link]:
        rows = self.conn.execute("SELECT * FROM links ORDER BY rowid").fetchall()
        return [Link(**dict(row)) for row in rows]

    def get_peer_links(self, path: str) -> list[Link]:
        rows = self.conn.execute(
            """
            SELECT *
            FROM links
            WHERE link_type = 'peer'
              AND (source_path = ? OR target_path = ?)
            ORDER BY rowid
            """,
            (path, path),
        ).fetchall()
        return [Link(**dict(row)) for row in rows]

    def get_peer_paths(self, path: str) -> list[str]:
        peer_paths: list[str] = []
        for link in self.get_peer_links(path):
            peer_path = link.target_path if link.source_path == path else link.source_path
            if peer_path not in peer_paths:
                peer_paths.append(peer_path)
        return peer_paths

    def get_chunks_by_path(self, path: str, include_children: bool = False) -> list[Chunk]:
        if include_children:
            rows = self.conn.execute(
                """
                SELECT *
                FROM chunks
                WHERE node_path = ? OR node_path LIKE ?
                ORDER BY rowid
                """,
                (path, f"{path}/%"),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM chunks WHERE node_path = ? ORDER BY rowid",
                (path,),
            ).fetchall()
        return [self._chunk_from_row(row) for row in rows]

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

    def _get_link_by_relationship(self, source_path: str, target_path: str, link_type: str) -> Link | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM links
            WHERE source_path = ? AND target_path = ? AND link_type = ?
            """,
            (source_path, target_path, link_type),
        ).fetchone()
        if row is None:
            return None
        return Link(**dict(row))

    def _chunk_from_row(self, row: sqlite3.Row) -> Chunk:
        payload: dict[str, Any] = dict(row)
        payload["lineage"] = json.loads(payload.pop("lineage_json"))
        return Chunk(**payload)
