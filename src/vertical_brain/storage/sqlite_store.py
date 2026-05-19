from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator
from uuid import uuid4

from vertical_brain.core.gold import gold_aspect_embed_key, parse_gold_content
from vertical_brain.core.models import Chunk, Link, Node, SearchResult, utc_now
from vertical_brain.core.namespace_map import normalize_namespace_root_path
from vertical_brain.storage.derivations import StorageDerivationsMixin

if TYPE_CHECKING:
    from vertical_brain.core.models import OperationResult, StorageOperation
from vertical_brain.core.search import fts_query, lexical_search, make_snippet, tokenize_query


# Serializes PRAGMA journal_mode=WAL across threads.
# SQLite's busy_timeout does not reliably apply to journal mode switching on
# all platforms/versions — a Python-level lock prevents the race entirely.
_WAL_INIT_LOCK = threading.Lock()

# Current on-disk schema version. Bump by one for every entry added to
# SQLiteStore._migration_steps so existing databases can be upgraded in order.
_SCHEMA_VERSION = 1


class SQLiteStore(StorageDerivationsMixin):
    VACUUM_INACTIVE_STATUSES = ("stale", "superseded", "legacy", "contradicted")
    VACUUM_MIN_RETENTION_HOURS = 168.0

    def __init__(self, root: str | Path = "data", db_name: str = "vertical_brain.sqlite"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_file = self.root / db_name
        self.namespace_roots_file = self.root / "namespaces" / "root.json"
        self._transaction_depth = 0

        self.conn = sqlite3.connect(self.db_file, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        # Serialize WAL mode switching: SQLite's busy_timeout does not reliably
        # protect PRAGMA journal_mode=WAL from concurrent "database is locked"
        # errors on all platforms/versions.  The Python lock ensures only one
        # thread sets WAL at a time; subsequent connections find WAL already set
        # and return immediately without needing a write lock.
        with _WAL_INIT_LOCK:
            self.conn.execute("PRAGMA busy_timeout=5000")
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._fts_enabled = self._init_search_index()
        if self._fts_enabled and self._search_index_needs_rebuild():
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
                is_dirty INTEGER NOT NULL DEFAULT 0,
                version INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS embedding_schema (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                model_name TEXT NOT NULL,
                vector_dimension INTEGER NOT NULL,
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
                valid_to TEXT,
                decay_factor REAL NOT NULL DEFAULT 1.0,
                metadata_json TEXT NOT NULL DEFAULT '{}'
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

            CREATE TABLE IF NOT EXISTS vector_cache (
                content_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (content_hash, model_name)
            );

            CREATE TABLE IF NOT EXISTS ingest_registry (
                content_hash TEXT PRIMARY KEY,
                source_namespace TEXT NOT NULL,
                file_name TEXT NOT NULL,
                ingested_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ingest_sessions (
                session_key TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                session_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        self.conn.commit()
        self._migrate_node_columns()
        self._migrate_chunk_columns()
        self._migrate_embedding_schema_columns()
        self._migrate_vector_cache_columns()
        # Composite dedup index — created after _migrate_chunk_columns so content_hash is guaranteed present.
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_dedup ON chunks(node_path, status, content_hash)"
        )
        self.conn.commit()
        self._apply_migrations()

    def _migration_steps(self) -> list[tuple[int, Callable[[], None]]]:
        """Ordered (target_version, step) pairs for upgrading tracked databases.

        Each step upgrades a database recorded at ``target_version - 1`` to
        ``target_version``. Version 1 is the baseline and needs no step.

        When the schema next changes structurally: update the CREATE TABLE
        baseline above, append a (2, self._migrate_v2) pair here, and bump
        _SCHEMA_VERSION to 2. A migration author adding the first step must
        also decide how an untracked legacy database (see _apply_migrations)
        is brought to version 1 before the step runs.
        """
        return []

    def _apply_migrations(self) -> None:
        """Bring the database up to _SCHEMA_VERSION via ordered migration steps.

        A database with no recorded version is assumed current — it was either
        just created by the CREATE TABLE baseline or already patched by the
        additive _migrate_*_columns helpers — and is simply stamped.
        """
        row = self.conn.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
        if row is None:
            # ON CONFLICT DO NOTHING: a second connection opening the same
            # database concurrently may insert the row between our SELECT and
            # INSERT — the upsert keeps that race harmless.
            self.conn.execute(
                "INSERT INTO schema_version (id, version, updated_at) VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO NOTHING",
                (_SCHEMA_VERSION, utc_now()),
            )
            self.conn.commit()
            return

        current = row["version"]
        for target, step in self._migration_steps():
            if current < target:
                step()
                current = target
        if current != row["version"]:
            self.conn.execute(
                "UPDATE schema_version SET version = ?, updated_at = ? WHERE id = 1",
                (current, utc_now()),
            )
            self.conn.commit()

    def schema_version(self) -> int:
        """Return the schema version recorded in the database."""
        row = self.conn.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
        return int(row["version"]) if row is not None else 0

    def _migrate_node_columns(self) -> None:
        existing = {row["name"] for row in self.conn.execute("PRAGMA table_info(nodes)").fetchall()}
        migrations = {
            "node_type": "ALTER TABLE nodes ADD COLUMN node_type TEXT NOT NULL DEFAULT 'default'",
            "gold_summary": "ALTER TABLE nodes ADD COLUMN gold_summary TEXT NOT NULL DEFAULT ''",
            "is_dirty": "ALTER TABLE nodes ADD COLUMN is_dirty INTEGER NOT NULL DEFAULT 0",
            "version": "ALTER TABLE nodes ADD COLUMN version INTEGER NOT NULL DEFAULT 0",
            "metadata_json": "ALTER TABLE nodes ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'",
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

    def _migrate_chunk_columns(self) -> None:
        existing = {row["name"] for row in self.conn.execute("PRAGMA table_info(chunks)").fetchall()}
        migrations = {
            "chunk_key": "ALTER TABLE chunks ADD COLUMN chunk_key TEXT",
            "content_hash": "ALTER TABLE chunks ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''",
            "supersedes": "ALTER TABLE chunks ADD COLUMN supersedes TEXT NOT NULL DEFAULT '[]'",
            "valid_from": "ALTER TABLE chunks ADD COLUMN valid_from TEXT NOT NULL DEFAULT ''",
            "valid_to": "ALTER TABLE chunks ADD COLUMN valid_to TEXT",
            "decay_factor": "ALTER TABLE chunks ADD COLUMN decay_factor REAL NOT NULL DEFAULT 1.0",
            "immutable": "ALTER TABLE chunks ADD COLUMN immutable INTEGER NOT NULL DEFAULT 0",
            "metadata_json": "ALTER TABLE chunks ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'",
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

    def _migrate_embedding_schema_columns(self) -> None:
        existing = {row["name"] for row in self.conn.execute("PRAGMA table_info(embedding_schema)").fetchall()}
        migrations = {
            "created_at": "ALTER TABLE embedding_schema ADD COLUMN created_at TEXT NOT NULL DEFAULT ''",
            "updated_at": "ALTER TABLE embedding_schema ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''",
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

    def _migrate_vector_cache_columns(self) -> None:
        existing = {row["name"] for row in self.conn.execute("PRAGMA table_info(vector_cache)").fetchall()}
        migrations = {
            "created_at": "ALTER TABLE vector_cache ADD COLUMN created_at TEXT NOT NULL DEFAULT ''",
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

    def _search_index_needs_rebuild(self) -> bool:
        """True only when chunks exist but the persistent FTS index is empty.

        The FTS index is a real table kept in sync incrementally by _index_chunk;
        rebuilding it on every connection open is pure waste (and a concurrency
        hazard). The one case that still needs a rebuild is an existing database
        that predates the FTS table — chunks present, index empty.
        """
        if self.conn.execute("SELECT EXISTS(SELECT 1 FROM search_index)").fetchone()[0]:
            return False
        return bool(self.conn.execute("SELECT EXISTS(SELECT 1 FROM chunks)").fetchone()[0])

    def _seed_namespace_roots(self) -> None:
        if not self.namespace_roots_file.exists():
            return

        payload = json.loads(self.namespace_roots_file.read_text(encoding="utf-8"))
        for root in payload.get("roots", []):
            if isinstance(root, str) and root:
                self.ensure_node(root)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Atomic unit of work, safely nestable.

        The outermost level drives a real BEGIN/COMMIT/ROLLBACK. Inner levels
        use SAVEPOINTs so a nested block can roll back its own writes
        independently — even if the surrounding level goes on to commit.
        """
        outermost = self._transaction_depth == 0
        savepoint = None if outermost else f"sp_{self._transaction_depth}"
        if outermost:
            self.conn.execute("BEGIN")
        else:
            self.conn.execute(f"SAVEPOINT {savepoint}")
        self._transaction_depth += 1
        try:
            yield
        except Exception:
            self._transaction_depth -= 1
            if outermost:
                self.conn.rollback()
            else:
                self.conn.execute(f"ROLLBACK TO {savepoint}")
                self.conn.execute(f"RELEASE {savepoint}")
            raise
        else:
            self._transaction_depth -= 1
            if outermost:
                self.conn.commit()
            else:
                self.conn.execute(f"RELEASE {savepoint}")

    def _commit_if_needed(self) -> None:
        if self._transaction_depth == 0:
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def checkpoint(self, mode: str = "PASSIVE") -> dict[str, int]:
        mode = mode.upper()
        if mode not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            raise ValueError(f"Unsupported checkpoint mode: {mode}")
        row = self.conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        return {
            "busy": int(row[0]),
            "log": int(row[1]),
            "checkpointed": int(row[2]),
        }

    def vacuum(
        self,
        *,
        retention_hours: float = VACUUM_MIN_RETENTION_HOURS,
        dry_run: bool = True,
        force: bool = False,
        include_immutable: bool = False,
        prune_empty_nodes: bool = True,
        prune_vector_cache: bool = True,
        reclaim_space: bool = False,
    ) -> dict[str, Any]:
        """Purge old inactive chunks and service indexes.

        This is the storage-maintenance equivalent of Databricks VACUUM:
        inactive chunks stay query-invisible immediately, then this command
        physically removes them after a retention window. Applying a retention
        below seven days requires *force*.
        """
        if retention_hours < 0:
            raise ValueError("retention_hours must be non-negative")
        if not dry_run and retention_hours < self.VACUUM_MIN_RETENTION_HOURS and not force:
            raise ValueError("Vacuum retention below 168 hours requires force=True")
        if not dry_run and include_immutable and not force:
            raise ValueError("Vacuuming immutable chunks requires force=True")

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=retention_hours)).isoformat()
        candidate_rows = self._vacuum_candidate_rows(cutoff, include_immutable=include_immutable)
        candidates = [
            {
                "id": row["id"],
                "path": row["node_path"],
                "layer": row["layer"],
                "content_type": row["content_type"],
                "status": row["status"],
                "source": row["source"],
                "valid_to": row["valid_to"],
                "updated_at": row["updated_at"],
            }
            for row in candidate_rows
        ]
        by_status: dict[str, int] = {}
        for candidate in candidates:
            status = candidate["status"]
            by_status[status] = by_status.get(status, 0) + 1

        result: dict[str, Any] = {
            "dry_run": dry_run,
            "retention_hours": retention_hours,
            "cutoff": cutoff,
            "statuses": list(self.VACUUM_INACTIVE_STATUSES),
            "include_immutable": include_immutable,
            "eligible_chunks": len(candidates),
            "eligible_by_status": by_status,
            "deleted_chunks": 0,
            "deleted_empty_nodes": 0,
            "deleted_vectors": 0,
            "reclaimed_space": False,
            "checkpoint": None,
            "candidates": candidates,
        }
        if dry_run:
            return result

        candidate_ids = [row["id"] for row in candidate_rows]
        with self.transaction():
            if candidate_ids:
                placeholders = ",".join("?" for _ in candidate_ids)
                if self._fts_enabled:
                    self.conn.execute(
                        f"""
                        DELETE FROM search_index
                        WHERE record_type = 'chunk'
                          AND record_id IN ({placeholders})
                        """,
                        candidate_ids,
                    )
                cursor = self.conn.execute(
                    f"DELETE FROM chunks WHERE id IN ({placeholders})",
                    candidate_ids,
                )
                result["deleted_chunks"] = cursor.rowcount

            if prune_vector_cache:
                active_keys = self._active_vector_cache_keys()
                if active_keys:
                    placeholders = ",".join("?" for _ in active_keys)
                    cursor = self.conn.execute(
                        f"DELETE FROM vector_cache WHERE content_hash NOT IN ({placeholders})",
                        list(active_keys),
                    )
                else:
                    cursor = self.conn.execute("DELETE FROM vector_cache")
                result["deleted_vectors"] = cursor.rowcount

            if prune_empty_nodes:
                result["deleted_empty_nodes"] = self._delete_empty_nodes()

        # search_index was pruned surgically above (DELETE … record_id IN candidates);
        # no full rebuild needed — node/vector pruning does not touch indexed chunk rows.
        if reclaim_space:
            self.conn.execute("VACUUM")
            result["reclaimed_space"] = True
        result["checkpoint"] = self.checkpoint("TRUNCATE")
        return result

    def _active_vector_cache_keys(self) -> set[str]:
        keys: set[str] = set()
        rows = self.conn.execute(
            """
            SELECT content_hash, layer, content
            FROM chunks
            WHERE status = 'active'
            """
        ).fetchall()
        for row in rows:
            if row["content_hash"]:
                keys.add(row["content_hash"])
            if row["layer"] == "gold":
                keys.update(
                    gold_aspect_embed_key(aspect)
                    for aspect in parse_gold_content(row["content"])
                    if aspect.strip()
                )
        return keys

    def backup_to(self, destination: str | Path, *, overwrite: bool = False) -> Path:
        if self._transaction_depth != 0:
            raise RuntimeError("Cannot create a SQLite backup while a transaction is open")

        backup_file = Path(destination)
        if backup_file.resolve() == self.db_file.resolve():
            raise ValueError("Backup destination must differ from the source database file")
        if backup_file.exists() and not overwrite:
            raise FileExistsError(f"Backup destination already exists: {backup_file}")

        self._commit_if_needed()
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = backup_file.with_name(f".{backup_file.name}.{uuid4().hex}.tmp")
        try:
            with sqlite3.connect(tmp_file) as target:
                self.conn.backup(target)
            tmp_file.replace(backup_file)
        except Exception:
            try:
                tmp_file.unlink()
            except FileNotFoundError:
                pass
            raise
        return backup_file

    def _vacuum_candidate_rows(self, cutoff: str, *, include_immutable: bool = False) -> list[sqlite3.Row]:
        placeholders = ",".join("?" for _ in self.VACUUM_INACTIVE_STATUSES)
        immutable_clause = "" if include_immutable else "AND immutable = 0"
        return self.conn.execute(
            f"""
            SELECT *
            FROM chunks
            WHERE status IN ({placeholders})
              AND COALESCE(NULLIF(valid_to, ''), updated_at, created_at) <= ?
              {immutable_clause}
            ORDER BY node_path ASC, created_at ASC
            """,
            (*self.VACUUM_INACTIVE_STATUSES, cutoff),
        ).fetchall()

    def _delete_empty_nodes(self) -> int:
        deleted = 0
        while True:
            cursor = self.conn.execute(
                """
                DELETE FROM nodes
                WHERE NOT EXISTS (
                    SELECT 1 FROM chunks WHERE chunks.node_path = nodes.path
                )
                  AND NOT EXISTS (
                    SELECT 1 FROM links
                    WHERE links.source_path = nodes.path
                       OR links.target_path = nodes.path
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM nodes AS child WHERE child.parent_path = nodes.path
                  )
                """
            )
            if cursor.rowcount == 0:
                return deleted
            deleted += cursor.rowcount

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
            row["is_dirty"] = 0
            row["version"] = node.version
            row["metadata_json"] = json.dumps(node.metadata or {}, ensure_ascii=False)
            row.pop("metadata", None)
            self.conn.execute(
                """
                INSERT INTO nodes (
                    id, path, name, parent_path, node_type, gold_summary, is_dirty, version,
                    created_at, updated_at, metadata_json
                )
                VALUES (
                    :id, :path, :name, :parent_path, :node_type, :gold_summary, :is_dirty, :version,
                    :created_at, :updated_at, :metadata_json
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
                chunk_key, content_hash, supersedes, valid_from, valid_to, decay_factor,
                immutable, metadata_json
            )
            VALUES (
                :id, :node_path, :content, :layer, :content_type, :status, :source,
                :confidence, :lineage_json, :created_at, :updated_at,
                :chunk_key, :content_hash, :supersedes, :valid_from, :valid_to, :decay_factor,
                :immutable, :metadata_json
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
                valid_to = :valid_to,
                decay_factor = :decay_factor,
                immutable = :immutable,
                metadata_json = :metadata_json
            WHERE id = :id
            """,
            row,
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Chunk not found: {chunk.id}")
        self._index_chunk(chunk)
        self._commit_if_needed()
        return chunk

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
        data["is_dirty"] = bool(data.get("is_dirty", 0))
        data.setdefault("version", 0)
        metadata_raw = data.pop("metadata_json", None)
        if isinstance(metadata_raw, str) and metadata_raw:
            try:
                parsed = json.loads(metadata_raw)
            except (TypeError, ValueError):
                parsed = {}
            data["metadata"] = parsed if isinstance(parsed, dict) else {}
        else:
            data["metadata"] = {}
        return data

    def update_node(self, node: Node) -> Node:
        cursor = self.conn.execute(
            """
            UPDATE nodes
            SET is_dirty = ?,
                version = ?,
                updated_at = ?,
                metadata_json = ?
            WHERE path = ?
            """,
            (
                1 if node.is_dirty else 0,
                node.version,
                utc_now(),
                json.dumps(node.metadata or {}, ensure_ascii=False),
                node.path,
            ),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Node not found: {node.path}")
        self._commit_if_needed()
        return node

    def bump_nodes_dirty(self, paths: list[str]) -> None:
        """Mark each existing node in *paths* dirty and increment its version by one.

        A single UPDATE replaces the per-ancestor read+write loop. Paths that do
        not correspond to a node are silently ignored (no row matches).
        """
        if not paths:
            return
        placeholders = ",".join("?" for _ in paths)
        self.conn.execute(
            f"""
            UPDATE nodes
            SET is_dirty = 1, version = version + 1, updated_at = ?
            WHERE path IN ({placeholders})
            """,
            (utc_now(), *paths),
        )
        self._commit_if_needed()

    def get_embedding_schema(self) -> dict | None:
        row = self.conn.execute("SELECT * FROM embedding_schema WHERE id = 1").fetchone()
        if row is None:
            return None
        return dict(row)

    def set_embedding_schema(self, model_name: str, vector_dimension: int) -> None:
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO embedding_schema (id, model_name, vector_dimension, created_at, updated_at)
            VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                model_name = excluded.model_name,
                vector_dimension = excluded.vector_dimension,
                updated_at = excluded.updated_at
            """,
            (model_name, vector_dimension, now, now),
        )
        self._commit_if_needed()

    def get_vector(self, content_hash: str, model_name: str) -> list[float] | None:
        row = self.conn.execute(
            "SELECT vector_json FROM vector_cache WHERE content_hash = ? AND model_name = ?",
            (content_hash, model_name),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def get_vectors(self, content_hashes: list[str], model_name: str) -> dict[str, list[float]]:
        """Batch-read cached vectors — one query per ~900 hashes instead of one per hash."""
        out: dict[str, list[float]] = {}
        unique = list(dict.fromkeys(h for h in content_hashes if h))
        batch_size = 900  # stay under SQLite's ~999 bound-parameter limit
        for start in range(0, len(unique), batch_size):
            batch = unique[start:start + batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = self.conn.execute(
                f"""
                SELECT content_hash, vector_json
                FROM vector_cache
                WHERE model_name = ? AND content_hash IN ({placeholders})
                """,
                (model_name, *batch),
            ).fetchall()
            for row in rows:
                out[row["content_hash"]] = json.loads(row["vector_json"])
        return out

    def set_vector(self, content_hash: str, model_name: str, vector: list[float]) -> None:
        self.conn.execute(
            """
            INSERT INTO vector_cache (content_hash, model_name, vector_json, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(content_hash, model_name) DO UPDATE SET
                vector_json = excluded.vector_json,
                created_at = excluded.created_at
            """,
            (content_hash, model_name, json.dumps(vector, ensure_ascii=False), utc_now()),
        )
        self._commit_if_needed()

    def delete_vectors_for_model(self, model_name: str) -> int:
        cursor = self.conn.execute(
            "DELETE FROM vector_cache WHERE model_name = ?", (model_name,)
        )
        self._commit_if_needed()
        return cursor.rowcount

    def rename_namespace(self, old_prefix: str, new_prefix: str) -> None:
        old_exact = old_prefix
        old_like = old_prefix + "/%"
        new_len = len(old_prefix)
        with self.transaction():
            self.conn.execute(
                """
                UPDATE chunks
                SET node_path = ? || SUBSTR(node_path, ?)
                WHERE node_path = ? OR node_path LIKE ?
                """,
                (new_prefix, new_len + 1, old_exact, old_like),
            )
            self.conn.execute(
                """
                UPDATE links
                SET source_path = ? || SUBSTR(source_path, ?)
                WHERE source_path = ? OR source_path LIKE ?
                """,
                (new_prefix, new_len + 1, old_exact, old_like),
            )
            self.conn.execute(
                """
                UPDATE links
                SET target_path = ? || SUBSTR(target_path, ?)
                WHERE target_path = ? OR target_path LIKE ?
                """,
                (new_prefix, new_len + 1, old_exact, old_like),
            )
            self.conn.execute(
                """
                UPDATE nodes
                SET path = ? || SUBSTR(path, ?)
                WHERE path = ? OR path LIKE ?
                """,
                (new_prefix, new_len + 1, old_exact, old_like),
            )
            self.conn.execute(
                """
                UPDATE nodes
                SET parent_path = ? || SUBSTR(parent_path, ?)
                WHERE parent_path = ? OR parent_path LIKE ?
                """,
                (new_prefix, new_len + 1, old_exact, old_like),
            )
            # Recalculate name and parent_path from the new path for all renamed nodes.
            renamed_rows = self.conn.execute(
                "SELECT path FROM nodes WHERE path = ? OR path LIKE ?",
                (new_prefix, new_prefix + "/%"),
            ).fetchall()
            for row in renamed_rows:
                new_path = row[0]
                parts = new_path.split("/")
                name = parts[-1]
                parent_path = "/".join(parts[:-1]) if len(parts) > 1 else None
                self.conn.execute(
                    "UPDATE nodes SET name = ?, parent_path = ? WHERE path = ?",
                    (name, parent_path, new_path),
                )
            # search_index.path is a stored column — rewrite it in place rather
            # than rebuilding the whole FTS index from scratch.
            if self._fts_enabled:
                self.conn.execute(
                    """
                    UPDATE search_index
                    SET path = ? || SUBSTR(path, ?)
                    WHERE path = ? OR path LIKE ?
                    """,
                    (new_prefix, new_len + 1, old_exact, old_like),
                )

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self.conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        return self._chunk_from_row(row) if row is not None else None

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

    def get_links_by_source(self, source_path: str) -> list[Link]:
        rows = self.conn.execute(
            "SELECT * FROM links WHERE source_path = ? ORDER BY rowid",
            (source_path,),
        ).fetchall()
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

    def has_active_chunk_with_hash(self, node_path: str, content_hash: str) -> bool:
        """Return True if an active chunk with the given content_hash exists at node_path.

        Uses the composite index (node_path, status, content_hash) — O(log n).
        """
        row = self.conn.execute(
            "SELECT 1 FROM chunks WHERE node_path = ? AND status = 'active' AND content_hash = ? LIMIT 1",
            (node_path, content_hash),
        ).fetchone()
        return row is not None

    def search(
        self,
        query: str,
        *,
        root_path: str | None = None,
        limit: int = 10,
        include_stale: bool = False,
    ) -> list[SearchResult]:
        root_path = normalize_namespace_root_path(root_path)
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

    # ------------------------------------------------------------------
    # Ingest registry
    # ------------------------------------------------------------------

    def register_ingest(self, content_hash: str, source_namespace: str, file_name: str) -> None:
        """Record a completed ingest so duplicate attempts can be detected."""
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO ingest_registry (content_hash, source_namespace, file_name, ingested_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(content_hash) DO UPDATE SET
                source_namespace=excluded.source_namespace,
                file_name=excluded.file_name,
                ingested_at=excluded.ingested_at
            """,
            (content_hash, source_namespace, file_name, now),
        )
        self.conn.commit()

    def find_ingest_by_hash(self, content_hash: str) -> dict[str, str] | None:
        """Return registry entry for content_hash, or None if not found."""
        row = self.conn.execute(
            "SELECT source_namespace, file_name, ingested_at FROM ingest_registry WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        if row is None:
            return None
        return {
            "source_namespace": row["source_namespace"],
            "file_name": row["file_name"],
            "ingested_at": row["ingested_at"],
        }

    # ------------------------------------------------------------------
    # Ingest sessions (persist across MCP restarts)
    # ------------------------------------------------------------------

    def save_ingest_session(self, session_key: str, state: str, session_json: str) -> None:
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO ingest_sessions (session_key, state, session_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_key) DO UPDATE SET
                state=excluded.state,
                session_json=excluded.session_json,
                updated_at=excluded.updated_at
            """,
            (session_key, state, session_json, now),
        )
        self.conn.commit()

    def get_ingest_session(self, session_key: str) -> dict[str, str] | None:
        row = self.conn.execute(
            "SELECT state, session_json, updated_at FROM ingest_sessions WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "state": row["state"],
            "session_json": row["session_json"],
            "updated_at": row["updated_at"],
        }

    def delete_ingest_session(self, session_key: str) -> None:
        self.conn.execute("DELETE FROM ingest_sessions WHERE session_key = ?", (session_key,))
        self.conn.commit()

    def list_ingest_sessions(self) -> list[dict[str, str]]:
        rows = self.conn.execute(
            "SELECT session_key, state, session_json, updated_at FROM ingest_sessions ORDER BY updated_at"
        ).fetchall()
        return [
            {
                "session_key": row["session_key"],
                "state": row["state"],
                "session_json": row["session_json"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

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
        row["immutable"] = int(chunk.immutable)
        row["metadata_json"] = json.dumps(chunk.metadata or {}, ensure_ascii=False)
        row.pop("metadata", None)
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
        payload.setdefault("decay_factor", 1.0)
        payload["immutable"] = bool(payload.get("immutable", 0))
        metadata_raw = payload.pop("metadata_json", None)
        if isinstance(metadata_raw, str) and metadata_raw:
            try:
                payload["metadata"] = json.loads(metadata_raw)
            except (TypeError, ValueError):
                payload["metadata"] = {}
        else:
            payload["metadata"] = {}
        if not isinstance(payload["metadata"], dict):
            payload["metadata"] = {}
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
