"""Ingest MCP handlers — mixin for VerticalBrainMCP."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vertical_brain.core.operations import StorageOperationExecutor
    from vertical_brain.storage.protocol import StorageProvider

from vertical_brain.core.models import Chunk, StorageOperation
from vertical_brain.mcp.extractors import extract_source_text as _extract_source_text
from vertical_brain.mcp.extractors.html import strip_html as _strip_html
from vertical_brain.mcp.ingest_protocol import (
    DEFAULT_INGEST_MODE,
    INGEST_MODE_ROUTING,
    build_protocol_lines,
    calculate_coverage_score,
    get_session_thresholds,
    inventory_probe_requirement,
    load_inventory_items,
    normalize_ingest_depth,
    normalize_ingest_mode,
    normalize_inventory_label,
    split_service_chunks,
    validate_inventory_probes,
    validate_namespace_ready_for_complete,
    validate_service_chunk_coverage,
    validate_skip_reason,
)


def _source_namespace(doc_slug: str) -> str:
    return f"SOURCES/{doc_slug}"


class _IngestHandlers:
    """Mixin providing ingest-related MCP handlers.

    Expects the host class to provide:
      self._store, self._executor, self._ingest_sessions, self._client_source
    """

    _store: "StorageProvider"
    _executor: "StorageOperationExecutor"
    _ingest_sessions: dict[str, dict[str, Any]]
    _client_source: str

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def _load_ingest_sessions_from_store(self) -> None:
        list_sessions = getattr(self._store, "list_ingest_sessions", None)
        if not callable(list_sessions):
            return
        for row in list_sessions():
            try:
                session = json.loads(row["session_json"])
            except json.JSONDecodeError:
                continue
            key = session.get("session_key") or row["session_key"]
            self._ingest_sessions[key] = session

    def _persist_ingest_session(self, session: dict[str, Any]) -> None:
        save = getattr(self._store, "save_ingest_session", None)
        if not callable(save):
            return
        save(
            session["session_key"],
            session.get("state", "EXTRACTING_BRONZE"),
            json.dumps(session, ensure_ascii=False),
        )

    def _delete_persisted_ingest_session(self, session_key: str) -> None:
        delete = getattr(self._store, "delete_ingest_session", None)
        if callable(delete):
            delete(session_key)

    def _cancel_ingest_sessions_for_namespace(self, source_namespace: str) -> int:
        keys_to_remove: set[str] = set()
        for key, session in self._ingest_sessions.items():
            if session.get("source_namespace") == source_namespace:
                keys_to_remove.add(key)

        list_sessions = getattr(self._store, "list_ingest_sessions", None)
        if callable(list_sessions):
            for row in list_sessions():
                key = row.get("session_key") or ""
                if not key:
                    continue
                try:
                    session = json.loads(row["session_json"])
                except json.JSONDecodeError:
                    keys_to_remove.add(key)
                    continue
                if session.get("source_namespace") == source_namespace:
                    keys_to_remove.add(key)

        for key in keys_to_remove:
            self._ingest_sessions.pop(key, None)
            self._delete_persisted_ingest_session(key)
        return len(keys_to_remove)

    def _require_ingest_session(self, session_key: str) -> dict[str, Any]:
        session = self._ingest_sessions.get(session_key)
        if session is not None:
            return session
        load = getattr(self._store, "get_ingest_session", None)
        if callable(load):
            row = load(session_key)
            if row is not None:
                session = json.loads(row["session_json"])
                self._ingest_sessions[session_key] = session
                return session
        raise ValueError(f"No active ingest session: {session_key!r}")

    def _wipe_source_namespace(self, path: str) -> int:
        get_chunks = getattr(self._store, "get_chunks_by_path", None)
        if not callable(get_chunks):
            return 0
        all_chunks = get_chunks(path, include_children=True)
        chunk_ids = [
            c.id for c in all_chunks
            if c.status == "active" and c.layer != "gold"
        ]
        if not chunk_ids:
            return 0
        id_to_path = {c.id: c.node_path for c in all_chunks}
        by_path: dict[str, list[str]] = defaultdict(list)
        for cid in chunk_ids:
            by_path[id_to_path[cid]].append(cid)
        marked = 0
        for node_path, ids in by_path.items():
            op = StorageOperation(
                operation="mark_stale",
                target_path=node_path,
                chunk_ids=ids,
                reasoning_summary="Auto-wipe before force re-ingest.",
                force_immutable=True,
            )
            self._executor.apply(op)
            marked += len(ids)
        return marked

    # ------------------------------------------------------------------
    # Session start
    # ------------------------------------------------------------------

    def _start_ingest_session(
        self,
        *,
        content: str,
        file_name: str,
        authority: str,
        doc_slug: str,
        ingest_mode: str,
        ingest_depth: str,
        force: bool,
        source_hash: str | None = None,
        source_size: int | None = None,
    ) -> str:
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        file_size = len(content.encode("utf-8"))
        source_ns = _source_namespace(doc_slug)

        find_ingest = getattr(self._store, "find_ingest_by_hash", None)
        if callable(find_ingest) and not force:
            existing = find_ingest(file_hash)
            if existing:
                raise ValueError(
                    f"This file was already ingested.\n"
                    f"  file: {existing['file_name']}\n"
                    f"  namespace: {existing['source_namespace']}\n"
                    f"  ingested_at: {existing['ingested_at']}\n"
                    f"  sha256: {file_hash}\n\n"
                    f"Pass force=true to re-ingest (wipe namespace first if replacing content)."
                )

        wiped = 0
        cancelled_sessions = 0
        if force:
            wiped = self._wipe_source_namespace(source_ns)
            cancelled_sessions = self._cancel_ingest_sessions_for_namespace(source_ns)

        session_key = secrets.token_hex(8)
        service_chunks = split_service_chunks(content, session_key)
        session: dict[str, Any] = {
            "session_key": session_key,
            "file_name": file_name,
            "authority": authority,
            "doc_slug": doc_slug,
            "source_namespace": source_ns,
            "content_hash": file_hash,
            "content_size": file_size,
            "source_hash": source_hash,
            "source_size": source_size,
            "ingest_mode": ingest_mode,
            "ingest_depth": ingest_depth,
            "state": "EXTRACTING_BRONZE",
            "chunks": service_chunks,
            "inventory_probes": [],
            "force_wiped": wiped,
        }
        self._ingest_sessions[session_key] = session
        self._persist_ingest_session(session)
        lines = build_protocol_lines(session)
        if force and wiped:
            lines += ["", f"force re-ingest: marked {wiped} prior chunk(s) stale under {source_ns}."]
        if cancelled_sessions > 0:
            lines += ["", f"cancelled_sessions: {cancelled_sessions}"]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _handle_ingest_file(self, args: dict[str, Any]) -> str:
        source_path = (args.get("source_path") or "").strip()
        source_hash: str | None = None
        source_size: int | None = None

        if source_path:
            content, source_hash, source_size = _extract_source_text(source_path)
            file_name = args.get("file_name") or os.path.basename(source_path)
        else:
            if "content" not in args:
                raise ValueError("ingest_file requires either source_path or content")
            if "file_name" not in args:
                raise ValueError("ingest_file requires file_name when content is supplied")
            content = args["content"]
            file_name = args["file_name"]
            ext = os.path.splitext(file_name)[1].lower()
            if ext in {".pdf", ".docx", ".doc", ".html", ".htm", ".odt"}:
                raise ValueError(
                    f"{ext} ingestion must use source_path so the server extracts the complete text. "
                    "Passing caller-supplied content for binary formats is rejected to prevent summarized payloads."
                )

        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        file_size = len(content.encode("utf-8"))
        expected_hash = (args.get("expected_sha256") or "").strip()
        if expected_hash and expected_hash != file_hash:
            raise ValueError(
                f"ingest_file integrity check failed: expected_sha256={expected_hash}, "
                f"actual_sha256={file_hash}"
            )
        expected_size = args.get("expected_size_bytes")
        if expected_size is not None and int(expected_size) != file_size:
            raise ValueError(
                f"ingest_file integrity check failed: expected_size_bytes={expected_size}, "
                f"actual_size_bytes={file_size}"
            )

        authority: str = (args.get("authority") or "").strip().replace(" ", "_")
        raw_slug = args.get("doc_slug") or os.path.splitext(file_name)[0]
        doc_slug = raw_slug.strip().replace(" ", "_").replace(".", "_")
        ingest_mode = normalize_ingest_mode(args.get("mode"))
        ingest_depth = normalize_ingest_depth(args.get("depth"))

        return self._start_ingest_session(
            content=content,
            file_name=file_name,
            authority=authority,
            doc_slug=doc_slug,
            ingest_mode=ingest_mode,
            ingest_depth=ingest_depth,
            force=bool(args.get("force", False)),
            source_hash=source_hash,
            source_size=source_size,
        )

    def _handle_ingest_url(self, args: dict[str, Any]) -> str:
        url: str = args["url"]
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Only http/https URLs are supported, got: {url!r}")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vertical-brain/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_bytes: bytes = resp.read()
                content_type: str = resp.headers.get_content_type() or ""
        except urllib.error.URLError as exc:
            raise ValueError(f"Failed to fetch URL: {exc}") from exc

        charset = "utf-8"
        if hasattr(resp.headers, "get_content_charset"):
            charset = resp.headers.get_content_charset() or "utf-8"
        try:
            raw_text = raw_bytes.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            raw_text = raw_bytes.decode("utf-8", errors="replace")

        content = _strip_html(raw_text) if "html" in content_type else raw_text

        parsed = urllib.parse.urlparse(url)
        authority: str = (args.get("authority") or parsed.hostname or "").strip().replace(" ", "_")
        path_parts = [p for p in parsed.path.split("/") if p]
        default_slug = path_parts[-1] if path_parts else parsed.hostname or "doc"
        raw_slug = args.get("doc_slug") or default_slug
        doc_slug = raw_slug.strip().replace(" ", "_").replace(".", "_")
        file_name = doc_slug or "url_document"
        ingest_mode = normalize_ingest_mode(args.get("mode"))
        ingest_depth = normalize_ingest_depth(args.get("depth"))

        return self._start_ingest_session(
            content=content,
            file_name=file_name,
            authority=authority,
            doc_slug=doc_slug,
            ingest_mode=ingest_mode,
            ingest_depth=ingest_depth,
            force=bool(args.get("force", False)),
        )

    def _handle_get_service_chunk(self, args: dict[str, Any]) -> str:
        session_key = args.get("session_key", "")
        chunk_id = args.get("chunk_id", "")
        session = self._require_ingest_session(session_key)
        chunk = next((c for c in session["chunks"] if c["id"] == chunk_id), None)
        if chunk is None:
            raise ValueError(f"Chunk {chunk_id!r} not found in session {session_key!r}")
        return json.dumps({
            "chunk_id": chunk_id,
            "index": chunk["index"],
            "chunk_type": chunk["chunk_type"],
            "status": chunk["status"],
            "content": chunk["content"],
        }, ensure_ascii=False)

    def _ingest_step_hint(self, session: dict[str, Any]) -> str:
        total = len(session.get("chunks") or [])
        processed = sum(1 for c in (session.get("chunks") or []) if c["status"] != "pending")
        pending = total - processed
        state = session.get("state", "")
        ns = session.get("source_namespace", "")
        if state == "BRONZE_COMPLETE":
            return "STEP 5 ✓ — Bronze complete. Next: STEP 6 — write Silver per sub-namespace, then submit_inventory_probes, complete_ingest."
        pct = int(processed / total * 100) if total else 0
        return (
            f"STEP 4 — Bronze extraction | {ns} | "
            f"{processed}/{total} chunks processed ({pct}%) | {pending} pending"
        )

    def _handle_get_service_chunks(self, args: dict[str, Any]) -> str:
        session_key = args.get("session_key", "")
        session = self._require_ingest_session(session_key)

        chunk_ids = args.get("chunk_ids") or []
        if chunk_ids:
            selected = []
            for chunk_id in chunk_ids[:10]:
                chunk = next((c for c in session["chunks"] if c["id"] == chunk_id), None)
                if chunk is None:
                    raise ValueError(f"Chunk {chunk_id!r} not found in session {session_key!r}")
                selected.append(chunk)
        else:
            start = int(args.get("start_index", 0))
            count = min(int(args.get("count", 5)), 10)
            if start < 0:
                raise ValueError("start_index must be >= 0")
            selected = session["chunks"][start: start + count]

        return json.dumps(
            {
                "session_key": session_key,
                "step": self._ingest_step_hint(session),
                "chunks": [
                    {
                        "chunk_id": c["id"],
                        "index": c["index"],
                        "chunk_type": c["chunk_type"],
                        "status": c["status"],
                        "content": c["content"],
                    }
                    for c in selected
                ],
            },
            ensure_ascii=False,
        )

    def _handle_mark_service_chunk(self, args: dict[str, Any]) -> str:
        session_key = args.get("session_key", "")
        chunk_id = args.get("chunk_id", "")
        status = args.get("status", "")
        skip_reason = args.get("skip_reason")
        if status not in ("extracted", "skipped"):
            raise ValueError(f"status must be 'extracted' or 'skipped', got {status!r}")
        if status == "skipped" and not skip_reason:
            raise ValueError("skip_reason is required when status='skipped'")
        session = self._require_ingest_session(session_key)
        if status == "skipped":
            validate_skip_reason(str(skip_reason), mode=session.get("ingest_mode", DEFAULT_INGEST_MODE))
        chunk = next((c for c in session["chunks"] if c["id"] == chunk_id), None)
        if chunk is None:
            raise ValueError(f"Chunk {chunk_id!r} not found in session {session_key!r}")
        chunk["status"] = status
        chunk["skip_reason"] = skip_reason
        self._persist_ingest_session(session)
        pending = sum(1 for c in session["chunks"] if c["status"] == "pending")
        return json.dumps(
            {"ok": True, "chunk_id": chunk_id, "status": status, "remaining_pending": pending},
            ensure_ascii=False,
        )

    def _handle_batch_mark_service_chunks(self, args: dict[str, Any]) -> str:
        session_key = args.get("session_key", "")
        marks = args.get("marks") or []
        if not marks:
            raise ValueError("marks must be a non-empty array")
        if len(marks) > 50:
            raise ValueError(f"batch_mark_service_chunks: max 50 marks per call, got {len(marks)}")
        session = self._require_ingest_session(session_key)
        ingest_mode = session.get("ingest_mode", DEFAULT_INGEST_MODE)
        chunk_index = {c["id"]: c for c in session["chunks"]}
        results = []
        for mark in marks:
            chunk_id = mark.get("chunk_id", "")
            status = mark.get("status", "")
            skip_reason = mark.get("skip_reason")
            if status not in ("extracted", "skipped"):
                raise ValueError(f"status must be 'extracted' or 'skipped', got {status!r} for chunk {chunk_id!r}")
            if status == "skipped" and not skip_reason:
                raise ValueError(f"skip_reason is required when status='skipped' (chunk {chunk_id!r})")
            if status == "skipped":
                validate_skip_reason(str(skip_reason), mode=ingest_mode)
            chunk = chunk_index.get(chunk_id)
            if chunk is None:
                raise ValueError(f"Chunk {chunk_id!r} not found in session {session_key!r}")
            chunk["status"] = status
            chunk["skip_reason"] = skip_reason
            results.append({"chunk_id": chunk_id, "status": status})
        self._persist_ingest_session(session)
        pending = sum(1 for c in session["chunks"] if c["status"] == "pending")
        return json.dumps(
            {
                "ok": True,
                "marked": len(results),
                "remaining_pending": pending,
                "step": self._ingest_step_hint(session),
                "results": results,
            },
            ensure_ascii=False,
        )

    def _handle_finish_bronze_extraction(self, args: dict[str, Any]) -> str:
        session_key = args.get("session_key", "")
        session = self._require_ingest_session(session_key)
        pending = [c for c in session["chunks"] if c["status"] == "pending"]
        if pending:
            ids = [c["id"] for c in pending[:5]]
            more = f" ... (+{len(pending) - 5} more)" if len(pending) > 5 else ""
            raise ValueError(
                f"{len(pending)} chunk(s) still pending — mark them extracted or skipped first. "
                f"Pending: {ids}{more}"
            )
        validate_service_chunk_coverage(session, phase="finish_bronze_extraction")
        extracted = sum(1 for c in session["chunks"] if c["status"] == "extracted")
        skipped = sum(1 for c in session["chunks"] if c["status"] == "skipped")
        session["state"] = "BRONZE_COMPLETE"
        self._persist_ingest_session(session)
        return json.dumps({
            "status": "ok",
            "step": "STEP 5 ✓ — finish_bronze_extraction passed.",
            "extracted": extracted,
            "skipped": skipped,
            "total": len(session["chunks"]),
            "next_action": (
                "STEP 6 — write Silver per sub-namespace as [chunk_id] one-liner index, "
                f"then root Silver at {session['source_namespace']}. "
                "STEP 6b — submit_inventory_probes. STEP 7 — complete_ingest."
            ),
        }, ensure_ascii=False)

    def _handle_submit_inventory_probes(self, args: dict[str, Any]) -> str:
        session_key = args.get("session_key", "")
        probes = args.get("probes") or []
        if not probes:
            raise ValueError("probes must be a non-empty list of {item, chunk_ids} objects")
        session = self._require_ingest_session(session_key)
        existing: list[dict] = list(session.get("inventory_probes") or [])
        existing_by_item: dict[str, dict] = {
            normalize_inventory_label(p["item"]): p for p in existing if p.get("item")
        }
        for probe in probes:
            item = probe.get("item", "")
            key = normalize_inventory_label(item)
            if key:
                existing_by_item[key] = probe
        session["inventory_probes"] = list(existing_by_item.values())
        self._persist_ingest_session(session)

        inventory_items = load_inventory_items(self._store, session["source_namespace"])
        inventory_count = len(inventory_items)
        thresholds = get_session_thresholds(session)
        required = inventory_probe_requirement(inventory_count, thresholds)

        total_accumulated = len(session["inventory_probes"])
        return json.dumps({
            "status": "ok",
            "submitted": total_accumulated,
            "required_for_complete": required,
            "inventory_item_count": inventory_count,
        }, ensure_ascii=False)

    def _handle_complete_ingest(self, args: dict[str, Any]) -> str:
        session_key = args.get("session_key", "")
        session = self._require_ingest_session(session_key)
        if session["state"] != "BRONZE_COMPLETE":
            raise ValueError(
                f"Cannot complete ingest: state is {session['state']!r}. "
                "Call finish_bronze_extraction first."
            )
        validate_service_chunk_coverage(session, phase="complete_ingest")
        storage_check = validate_namespace_ready_for_complete(self._store, session)
        probe_check = validate_inventory_probes(session, self._store)
        coverage = calculate_coverage_score(session, self._store)
        extracted = sum(1 for c in session["chunks"] if c["status"] == "extracted")
        skipped = sum(1 for c in session["chunks"] if c["status"] == "skipped")
        source_ns = session["source_namespace"]
        ingest_mode = session.get("ingest_mode", DEFAULT_INGEST_MODE)

        if ingest_mode == INGEST_MODE_ROUTING:
            self._store.save_chunk(Chunk(
                node_path=source_ns,
                layer="bronze",
                content_type="note",
                content=(
                    "[ROUTING ONLY] Discoverability ingest — not answer-complete. "
                    "Silver contains section index only. "
                    "Do not cite this namespace for authoritative answers."
                ),
            ))

        register_ingest = getattr(self._store, "register_ingest", None)
        if callable(register_ingest):
            register_ingest(session["content_hash"], source_ns, session["file_name"])

        del self._ingest_sessions[session_key]
        self._delete_persisted_ingest_session(session_key)
        return json.dumps({
            "status": "completed",
            "source_namespace": source_ns,
            "ingest_mode": ingest_mode,
            "bronze_extracted": extracted,
            "service_chunks_skipped": skipped,
            "storage_check": storage_check,
            "inventory_probes": probe_check,
            "coverage_score": coverage,
            "session_key": session_key,
            "next_steps": (
                "MANDATORY: call append_gold_aspect for the source namespace AND each "
                "sub-namespace to add short routing tags (30-100 chars each). "
                "Gold is how future agents discover this content via session_start. "
                "Without Gold, the document is invisible to routing."
            ),
        }, ensure_ascii=False)
