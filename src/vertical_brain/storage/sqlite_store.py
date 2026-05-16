from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator
from uuid import uuid4

from vertical_brain.core.models import Chunk, Link, Node, SearchResult, utc_now

if TYPE_CHECKING:
    from vertical_brain.core.models import OperationResult, StorageOperation
from vertical_brain.core.search import fts_query, lexical_search, make_snippet, tokenize_query


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
        self._fts_enabled = self._init_search_index()
        if self._fts_enabled:
            self.rebuild_search_index()
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
                updated_at TEXT NOT NULL,
                chunk_key TEXT,
                content_hash TEXT NOT NULL DEFAULT '',
                supersedes TEXT NOT NULL DEFAULT '[]',
                valid_from TEXT NOT NULL DEFAULT '',
                valid_to TEXT
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

            CREATE TABLE IF NOT EXISTS operation_audit (
                id TEXT PRIMARY KEY,
                operation_type TEXT NOT NULL,
                target_path TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                status TEXT NOT NULL,
                reasoning_summary TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        self.conn.commit()
        self._migrate_chunk_columns()

    def _migrate_chunk_columns(self) -> None:
        existing = {row["name"] for row in self.conn.execute("PRAGMA table_info(chunks)").fetchall()}
        migrations = {
            "chunk_key": "ALTER TABLE chunks ADD COLUMN chunk_key TEXT",
            "content_hash": "ALTER TABLE chunks ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''",
            "supersedes": "ALTER TABLE chunks ADD COLUMN supersedes TEXT NOT NULL DEFAULT '[]'",
            "valid_from": "ALTER TABLE chunks ADD COLUMN valid_from TEXT NOT NULL DEFAULT ''",
            "valid_to": "ALTER TABLE chunks ADD COLUMN valid_to TEXT",
        }
        changed = False
        for column, ddl in migrations.items():
            if column not in existing:
                try:
                    self.conn.execute(ddl)
                    changed = True
                except sqlite3.OperationalError:
                    pass
        if changed:
            self.conn.commit()

    def _init_search_index(self) -> bool:
        try:
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
                    record_type UNINDEXED,
                    record_id UNINDEXED,
                    path,
                    layer UNINDEXED,
                    content_type UNINDEXED,
                    status UNINDEXED,
                    chunk_source UNINDEXED,
                    content,
                    tokenize='unicode61'
                )
                """
            )
        except sqlite3.OperationalError:
            return False
        self.conn.commit()
        return True

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
            row = asdict(node)
            row["gold_summary"] = ""
            self.conn.execute(
                """
                INSERT INTO nodes (
                    id, path, name, parent_path, node_type, gold_summary, created_at, updated_at
                )
                VALUES (
                    :id, :path, :name, :parent_path, :node_type, :gold_summary, :created_at, :updated_at
                )
                """,
                row,
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
        row = self._chunk_row(chunk)
        self.conn.execute(
            """
            INSERT INTO chunks (
                id, node_path, content, layer, content_type, status, source,
                confidence, lineage_json, created_at, updated_at,
                chunk_key, content_hash, supersedes, valid_from, valid_to
            )
            VALUES (
                :id, :node_path, :content, :layer, :content_type, :status, :source,
                :confidence, :lineage_json, :created_at, :updated_at,
                :chunk_key, :content_hash, :supersedes, :valid_from, :valid_to
            )
            """,
            row,
        )
        self._index_chunk(chunk)
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
        row = self._chunk_row(chunk)
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
                updated_at = :updated_at,
                chunk_key = :chunk_key,
                content_hash = :content_hash,
                supersedes = :supersedes,
                valid_from = :valid_from,
                valid_to = :valid_to
            WHERE id = :id
            """,
            row,
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Chunk not found: {chunk.id}")
        self._index_chunk(chunk)
        self._commit_if_needed()
        return chunk

    def gold_summary_path(self, path: str) -> Path:
        parts = path.split("/")
        return self.gold_dir.joinpath(*parts).with_suffix(".md")

    def get_node(self, path: str) -> Node | None:
        row = self.conn.execute("SELECT * FROM nodes WHERE path = ?", (path,)).fetchone()
        if row is None:
            return None
        return Node(**self._node_data_from_row(row))

    def list_nodes(self) -> list[Node]:
        rows = self.conn.execute("SELECT * FROM nodes ORDER BY rowid").fetchall()
        return [Node(**self._node_data_from_row(row)) for row in rows]

    def _node_data_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data.pop("gold_summary", None)
        return data

    def list_chunks(self) -> list[Chunk]:
        rows = self.conn.execute("SELECT * FROM chunks ORDER BY rowid").fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def list_links(self) -> list[Link]:
        rows = self.conn.execute("SELECT * FROM links ORDER BY rowid").fetchall()
        return [Link(**dict(row)) for row in rows]

    def get_link(self, link_id: str) -> Link | None:
        row = self.conn.execute("SELECT * FROM links WHERE id = ?", (link_id,)).fetchone()
        if row is None:
            return None
        return Link(**dict(row))

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

    def search(
        self,
        query: str,
        *,
        root_path: str | None = None,
        limit: int = 10,
        include_stale: bool = False,
    ) -> list[SearchResult]:
        terms = tokenize_query(query)
        if not terms or limit <= 0:
            return []
        if not self._fts_enabled:
            return lexical_search(
                chunks=self.list_chunks(),
                query=query,
                root_path=root_path,
                limit=limit,
                include_stale=include_stale,
            )

        where = ["search_index MATCH ?"]
        params: list[Any] = [fts_query(query)]
        if root_path:
            where.append("(path = ? OR path LIKE ?)")
            params.extend([root_path, f"{root_path}/%"])
        if not include_stale:
            where.append("status = 'active'")
        params.append(limit)

        try:
            rows = self.conn.execute(
                f"""
                SELECT
                    record_type,
                    record_id,
                    path,
                    layer,
                    content_type,
                    status,
                    content,
                    bm25(search_index) AS rank
                FROM search_index
                WHERE {" AND ".join(where)}
                ORDER BY rank ASC, path ASC, record_type ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
        except sqlite3.OperationalError:
            return lexical_search(
                nodes=self.list_nodes(),
                chunks=self.list_chunks(),
                query=query,
                root_path=root_path,
                limit=limit,
                include_stale=include_stale,
            )

        return [
            SearchResult(
                path=row["path"],
                source=row["record_type"],
                score=round(max(0.0, -float(row["rank"])), 6),
                snippet=make_snippet(row["content"], terms),
                chunk_id=row["record_id"] if row["record_type"] == "chunk" else None,
                layer=row["layer"],
                content_type=row["content_type"],
                status=row["status"],
            )
            for row in rows
        ]

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

    def rebuild_search_index(self) -> None:
        if not self._fts_enabled:
            return

        self.conn.execute("DELETE FROM search_index")
        for chunk in self.list_chunks():
            self._index_chunk(chunk)
        # Gold is now indexed as regular chunks (layer="gold")
        self._commit_if_needed()

    def log_audit(self, operation: "StorageOperation", result: "OperationResult") -> None:
        self.conn.execute(
            """
            INSERT INTO operation_audit (
                id, operation_type, target_path, payload_json, result_json,
                status, reasoning_summary, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                operation.operation,
                operation.target_path,
                operation.to_json(),
                result.to_json(),
                result.status,
                operation.reasoning_summary,
                utc_now(),
            ),
        )
        self._commit_if_needed()

    def list_audit(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM operation_audit ORDER BY rowid"
        ).fetchall()
        return [dict(row) for row in rows]

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

    def _chunk_row(self, chunk: Chunk) -> dict[str, Any]:
        row = asdict(chunk)
        row["lineage_json"] = json.dumps(chunk.lineage, ensure_ascii=False)
        del row["lineage"]
        row["supersedes"] = json.dumps(chunk.supersedes, ensure_ascii=False)
        return row

    def _chunk_from_row(self, row: sqlite3.Row) -> Chunk:
        payload: dict[str, Any] = dict(row)
        payload["lineage"] = json.loads(payload.pop("lineage_json"))
        supersedes = payload.get("supersedes")
        if isinstance(supersedes, str):
            payload["supersedes"] = json.loads(supersedes) if supersedes else []
        elif supersedes is None:
            payload["supersedes"] = []
        if not payload.get("valid_from"):
            payload["valid_from"] = payload.get("created_at") or ""
        if payload.get("content_hash") is None:
            payload["content_hash"] = ""
        return Chunk(**payload)

    def _index_chunk(self, chunk: Chunk) -> None:
        if not self._fts_enabled:
            return

        self.conn.execute(
            "DELETE FROM search_index WHERE record_type = 'chunk' AND record_id = ?",
            (chunk.id,),
        )
        self.conn.execute(
            """
            INSERT INTO search_index (
                record_type,
                record_id,
                path,
                layer,
                content_type,
                status,
                chunk_source,
                content
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "chunk",
                chunk.id,
                chunk.node_path,
                chunk.layer,
                chunk.content_type,
                chunk.status,
                chunk.source,
                chunk.content,
            ),
        )

