from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from vertical_brain.core.models import STAGING_PATH

if TYPE_CHECKING:
    from vertical_brain.storage.protocol import StorageProvider


STAGING_AGE_WARNING_DAYS = 3


_VALID_LAYERS = {"bronze", "silver", "gold"}
_VALID_CONTENT_TYPES = {
    "fact", "correction", "decision", "question", "note", "code", "artifact",
}
_VALID_STATUSES = {
    "active", "stale", "legacy", "superseded", "contradicted", "uncertain",
}


@dataclass
class DoctorIssue:
    severity: str  # "error" | "warning"
    check: str
    message: str
    path: str | None = None


class Doctor:
    """Runs integrity checks against a storage backend."""

    def __init__(self, store: "StorageProvider") -> None:
        self._store = store

    def run(self) -> list[DoctorIssue]:
        issues: list[DoctorIssue] = []
        issues += self._check_orphan_links()
        issues += self._check_chunks_missing_nodes()
        issues += self._check_duplicate_active_chunks_by_hash()
        issues += self._check_broken_gold_overflow_chains()
        issues += self._check_invalid_namespace_paths()
        issues += self._check_empty_chunk_content()
        issues += self._check_invalid_chunk_fields()
        issues += self._check_links_missing_reason()
        issues += self._check_fts_index_stale_leak()
        issues += self._check_stale_staging_chunks()
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
        by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
        for chunk in self._store.list_chunks():
            if chunk.status != "active":
                continue
            dedup_key = chunk.content_hash or chunk.content
            by_key[(chunk.node_path, dedup_key)].append(chunk.id)
        for (path, _key), chunk_ids in by_key.items():
            if len(chunk_ids) > 1:
                issues.append(DoctorIssue(
                    severity="warning",
                    check="duplicate_active_chunk",
                    message=f"{len(chunk_ids)} active chunks share the same dedupe key",
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
            SELECT s.record_id, c.status
            FROM search_index s
            JOIN chunks c ON s.record_id = c.id
            WHERE s.record_type = 'chunk' AND c.status != 'active'
            """
        ).fetchall()
        for row in rows:
            issues.append(DoctorIssue(
                severity="warning",
                check="fts_stale_leak",
                message=f"chunk {row[0]} has status '{row[1]}' but is still in the FTS search index",
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
