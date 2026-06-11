from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from vertical_brain.core.gold import parse_gold_content
from vertical_brain.core.models import STAGING_PATH
# Overload thresholds are owned by the executor — import them so doctor's
# advice and the executor's silver_too_large / gold_near_limit signals agree.
from vertical_brain.core.operations import _GOLD_NEAR_LIMIT, _SILVER_MAX_CHARS
from vertical_brain.core.vector_lsh import cluster_by_similarity

if TYPE_CHECKING:
    from vertical_brain.llm.embedding import EmbeddingProvider
    from vertical_brain.storage.protocol import StorageProvider


STAGING_AGE_WARNING_DAYS = 3

# A namespace needs at least this many active Bronze chunks before a split is
# worth suggesting, and its Bronze must fall into at least this many sizeable
# clusters. Similarity above this cosine threshold puts two chunks in one cluster.
_MIN_BRONZE_FOR_SPLIT = 4
_MIN_SPLIT_CLUSTERS = 2
_SPLIT_CLUSTER_THRESHOLD = 0.5


_VALID_LAYERS = {"bronze", "silver", "gold"}
_VALID_CONTENT_TYPES = {
    "fact", "reference", "correction", "decision", "question", "note", "code",
    "artifact",
}
_VALID_STATUSES = {
    "active", "stale", "legacy", "superseded", "contradicted", "uncertain",
}


@dataclass
class DoctorIssue:
    severity: str  # "error" | "warning" | "info"
    check: str
    message: str
    path: str | None = None


class Doctor:
    """Runs integrity checks against a storage backend."""

    def __init__(
        self,
        store: "StorageProvider",
        embedding_provider: "EmbeddingProvider | None" = None,
    ) -> None:
        self._store = store
        self._embedding_provider = embedding_provider

    def run(self) -> list[DoctorIssue]:
        issues: list[DoctorIssue] = []
        issues += self._check_orphan_links()
        issues += self._check_chunks_missing_nodes()
        issues += self._check_duplicate_active_chunks_by_hash()
        issues += self._check_multiple_active_silver()
        issues += self._check_broken_gold_overflow_chains()
        issues += self._check_invalid_namespace_paths()
        issues += self._check_empty_chunk_content()
        issues += self._check_invalid_chunk_fields()
        issues += self._check_links_missing_reason()
        issues += self._check_fts_index_stale_leak()
        issues += self._check_stale_staging_chunks()
        issues += self._check_namespace_overloaded()
        return issues

    def _node_paths(self) -> set[str]:
        return {node.path for node in self._store.list_nodes()}

    def _check_orphan_links(self) -> list[DoctorIssue]:
        issues: list[DoctorIssue] = []
        node_paths = self._node_paths()
        for link in self._store.list_links():
            if link.source_path not in node_paths:
                issues.append(DoctorIssue(
                    severity="error",
                    check="orphan_link",
                    message=f"link {link.id} source_path is not a known node",
                    path=link.source_path,
                ))
            if link.target_path not in node_paths:
                issues.append(DoctorIssue(
                    severity="error",
                    check="orphan_link",
                    message=f"link {link.id} target_path is not a known node",
                    path=link.target_path,
                ))
        return issues

    def _check_chunks_missing_nodes(self) -> list[DoctorIssue]:
        issues: list[DoctorIssue] = []
        node_paths = self._node_paths()
        for chunk in self._store.list_chunks():
            if chunk.node_path not in node_paths:
                issues.append(DoctorIssue(
                    severity="error",
                    check="chunk_missing_node",
                    message=f"chunk {chunk.id} references an unknown node",
                    path=chunk.node_path,
                ))
        return issues

    def _check_duplicate_active_chunks_by_hash(self) -> list[DoctorIssue]:
        issues: list[DoctorIssue] = []
        by_key: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for chunk in self._store.list_chunks():
            if chunk.status != "active":
                continue
            dedup_key = chunk.content_hash or chunk.content
            # Bronze evidence and Silver summary can intentionally have the same
            # text. Only same-layer duplicates are redundant active chunks.
            by_key[(chunk.node_path, chunk.layer, dedup_key)].append(chunk.id)
        for (path, _layer, _key), chunk_ids in by_key.items():
            if len(chunk_ids) > 1:
                issues.append(DoctorIssue(
                    severity="warning",
                    check="duplicate_active_chunk",
                    message=f"{len(chunk_ids)} active chunks share the same dedupe key",
                    path=path,
                ))
        return issues

    def _check_multiple_active_silver(self) -> list[DoctorIssue]:
        issues: list[DoctorIssue] = []
        by_path: dict[str, list[str]] = defaultdict(list)
        for chunk in self._store.list_chunks():
            if chunk.layer == "silver" and chunk.status == "active":
                by_path[chunk.node_path].append(chunk.id)
        for path, chunk_ids in by_path.items():
            if len(chunk_ids) > 1:
                issues.append(DoctorIssue(
                    severity="warning",
                    check="multiple_active_silver",
                    message=(
                        f"{len(chunk_ids)} active Silver chunks found; "
                        "use update_silver to keep one current summary"
                    ),
                    path=path,
                ))
        return issues

    def _check_broken_gold_overflow_chains(self) -> list[DoctorIssue]:
        issues: list[DoctorIssue] = []
        node_paths = self._node_paths()
        for link in self._store.list_links():
            if link.link_type == "gold_overflow" and link.target_path not in node_paths:
                issues.append(DoctorIssue(
                    severity="error",
                    check="broken_gold_overflow",
                    message=f"gold_overflow link {link.id} points to a missing node",
                    path=link.target_path,
                ))
        return issues

    def _check_invalid_namespace_paths(self) -> list[DoctorIssue]:
        issues: list[DoctorIssue] = []
        for node in self._store.list_nodes():
            path = node.path
            if path.startswith("/") or path.endswith("/") or "//" in path:
                issues.append(DoctorIssue(
                    severity="error",
                    check="invalid_namespace_path",
                    message="namespace path has a leading/trailing slash or empty segment",
                    path=path,
                ))
        return issues

    def _check_empty_chunk_content(self) -> list[DoctorIssue]:
        issues: list[DoctorIssue] = []
        for chunk in self._store.list_chunks():
            if not chunk.content or not chunk.content.strip():
                issues.append(DoctorIssue(
                    severity="error",
                    check="empty_chunk_content",
                    message=f"chunk {chunk.id} has empty content",
                    path=chunk.node_path,
                ))
        return issues

    def _check_invalid_chunk_fields(self) -> list[DoctorIssue]:
        issues: list[DoctorIssue] = []
        for chunk in self._store.list_chunks():
            if chunk.layer not in _VALID_LAYERS:
                issues.append(DoctorIssue(
                    severity="error",
                    check="invalid_chunk_field",
                    message=f"chunk {chunk.id} has invalid layer '{chunk.layer}'",
                    path=chunk.node_path,
                ))
            if chunk.content_type not in _VALID_CONTENT_TYPES:
                issues.append(DoctorIssue(
                    severity="error",
                    check="invalid_chunk_field",
                    message=f"chunk {chunk.id} has invalid content_type '{chunk.content_type}'",
                    path=chunk.node_path,
                ))
            if chunk.status not in _VALID_STATUSES:
                issues.append(DoctorIssue(
                    severity="error",
                    check="invalid_chunk_field",
                    message=f"chunk {chunk.id} has invalid status '{chunk.status}'",
                    path=chunk.node_path,
                ))
        return issues

    def _check_links_missing_reason(self) -> list[DoctorIssue]:
        issues: list[DoctorIssue] = []
        for link in self._store.list_links():
            if not link.reason or not link.reason.strip():
                issues.append(DoctorIssue(
                    severity="warning",
                    check="link_missing_reason",
                    message=f"link {link.id} has an empty reason",
                    path=link.source_path,
                ))
            if not link.link_type or not link.link_type.strip():
                issues.append(DoctorIssue(
                    severity="error",
                    check="link_missing_reason",
                    message=f"link {link.id} has an empty link_type",
                    path=link.source_path,
                ))
        return issues

    def _check_fts_index_stale_leak(self) -> list[DoctorIssue]:
        conn = getattr(self._store, "conn", None)
        fts_enabled = getattr(self._store, "_fts_enabled", False)
        if conn is None or not fts_enabled:
            return []
        issues: list[DoctorIssue] = []
        rows = conn.execute(
            """
            SELECT s.record_id, s.status AS index_status, c.status AS chunk_status
            FROM search_index s
            JOIN chunks c ON s.record_id = c.id
            WHERE s.record_type = 'chunk' AND s.status != c.status
            """
        ).fetchall()
        for row in rows:
            issues.append(DoctorIssue(
                severity="warning",
                check="fts_stale_leak",
                message=(
                    f"chunk {row[0]} has status '{row['chunk_status']}' "
                    f"but FTS index stores '{row['index_status']}'"
                ),
                path=None,
            ))
        return issues

    def _check_stale_staging_chunks(self) -> list[DoctorIssue]:
        issues: list[DoctorIssue] = []
        now = datetime.now(timezone.utc)
        for chunk in self._store.list_chunks():
            if chunk.node_path != STAGING_PATH or chunk.status != "active":
                continue
            try:
                created = datetime.fromisoformat(chunk.created_at)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = (now - created).total_seconds() / 86400
            except (ValueError, TypeError):
                age_days = 0.0
            if age_days >= STAGING_AGE_WARNING_DAYS:
                issues.append(DoctorIssue(
                    severity="warning",
                    check="stale_staging_chunk",
                    message=(
                        f"chunk {chunk.id} has been in {STAGING_PATH} for "
                        f"{age_days:.0f} days without reclassification"
                    ),
                    path=STAGING_PATH,
                ))
        return issues

    def _check_namespace_overloaded(self) -> list[DoctorIssue]:
        """Advisory: an overloaded namespace whose Bronze splits into distinct
        topic clusters should be decomposed into sub-namespaces.

        Skipped without an embedding provider — clustering needs vectors.
        """
        if self._embedding_provider is None:
            return []

        by_node: dict[str, list] = defaultdict(list)
        for chunk in self._store.list_chunks():
            if chunk.status == "active":
                by_node[chunk.node_path].append(chunk)

        issues: list[DoctorIssue] = []
        for path, chunks in sorted(by_node.items()):
            silver_len = max(
                (len(c.content) for c in chunks if c.layer == "silver"), default=0
            )
            gold_count = max(
                (len(parse_gold_content(c.content)) for c in chunks if c.layer == "gold"),
                default=0,
            )
            if silver_len <= _SILVER_MAX_CHARS and gold_count < _GOLD_NEAR_LIMIT:
                continue  # not overloaded — no split advice

            bronze = [c for c in chunks if c.layer == "bronze"]
            if len(bronze) < _MIN_BRONZE_FOR_SPLIT:
                continue

            vectors: dict[str, list[float]] = {}
            for chunk in bronze:
                vec = self._embed(chunk)
                if vec is not None:
                    vectors[chunk.id] = vec
            if len(vectors) < _MIN_BRONZE_FOR_SPLIT:
                continue

            clusters = cluster_by_similarity(vectors, threshold=_SPLIT_CLUSTER_THRESHOLD)
            sizeable = [group for group in clusters if len(group) >= 2]
            if len(sizeable) >= _MIN_SPLIT_CLUSTERS:
                issues.append(DoctorIssue(
                    severity="info",
                    check="namespace_overloaded",
                    message=(
                        f"namespace is overloaded and its Bronze splits into "
                        f"{len(sizeable)} topic clusters — consider decomposing it "
                        f"into sub-namespaces, each with its own focused Silver"
                    ),
                    path=path,
                ))
        return issues

    def _embed(self, chunk) -> "list[float] | None":  # type: ignore[no-untyped-def]
        """Embed a chunk, preferring a cached vector to avoid network calls."""
        assert self._embedding_provider is not None
        model_name = getattr(self._embedding_provider, "model_name", None)
        get_vector = getattr(self._store, "get_vector", None)
        if model_name and callable(get_vector):
            cached = get_vector(chunk.content_hash, model_name)
            if isinstance(cached, list) and cached:
                return cached
        vec = self._embedding_provider.embed(chunk.content)
        return vec if isinstance(vec, list) and vec else None
