"""Ingest session protocol, splitting, and server-side enforcement."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vertical_brain.core.models import Chunk
    from vertical_brain.storage.protocol import StorageProvider

INGEST_MODE_ANSWER_COMPLETE = "answer_complete"
INGEST_MODE_ROUTING = "routing"
DEFAULT_INGEST_MODE = INGEST_MODE_ANSWER_COMPLETE

# answer_complete: document must be answerable without the source file
MAX_SKIP_RATIO_ANSWER_COMPLETE = 0.25
MIN_EXTRACTED_RATIO_ANSWER_COMPLETE = 0.50
MIN_BRONZE_FACTS_ANSWER_COMPLETE = 1

INVENTORY_PREFIX = "[INVENTORY]"

INVALID_SKIP_REASON_FRAGMENTS = (
    "bulk skip",
    "mass skip",
    "skip all",
    "skip remaining",
    "skip rest",
    "complete_ingest",
    "finish_bronze_extraction",
    "close ingest",
    "close the ingest",
    "clean re-ingest verification",
    "no time",
    "too long",
    "too many",
    "finish session",
    "close session",
)

WEAK_SKIP_REASONS = frozenset({
    "test",
    "n/a",
    "na",
    "skip",
    "none",
    "misc",
    "other",
    "todo",
    "tbd",
    "...",
})


def normalize_ingest_mode(mode: str | None) -> str:
    value = (mode or DEFAULT_INGEST_MODE).strip().lower()
    if value not in (INGEST_MODE_ANSWER_COMPLETE, INGEST_MODE_ROUTING):
        raise ValueError(
            f"ingest mode must be {INGEST_MODE_ANSWER_COMPLETE!r} or {INGEST_MODE_ROUTING!r}, got {mode!r}"
        )
    return value


def validate_skip_reason(skip_reason: str, *, mode: str) -> None:
    normalized = str(skip_reason).strip().lower()
    if len(normalized) < 12:
        raise ValueError(
            "skip_reason must be at least 12 characters and name the specific boilerplate "
            "(e.g. 'page footer copyright only, no payment data')."
        )
    if normalized in WEAK_SKIP_REASONS:
        raise ValueError(
            f"skip_reason {skip_reason!r} is too vague. Name what was skipped and why it is not answer-critical."
        )
    if any(fragment in normalized for fragment in INVALID_SKIP_REASON_FRAGMENTS):
        raise ValueError(
            "skip_reason describes completing or bulk-closing ingest, not a content reason. "
            "Use skipped only for empty/formatting noise, boilerplate, irrelevant text, or duplicates."
        )
    if mode == INGEST_MODE_ANSWER_COMPLETE and any(
        word in normalized
        for word in ("summar", "abbrev", "overview only", "not important", "low priority")
    ):
        raise ValueError(
            "In answer_complete mode you cannot skip substantive source knowledge. "
            "Extract facts or report that ingest cannot finish yet."
        )


def split_service_chunks(content: str, session_key: str) -> list[dict[str, Any]]:
    """Split extracted text into service chunks for guided ingest."""
    segments: list[tuple[str, str]] = []
    fence_re = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)
    last_end = 0

    for match in fence_re.finditer(content):
        before = content[last_end : match.start()]
        if before.strip():
            segments.extend(_paragraph_segments(before))
        segments.append((match.group(0).strip(), "atomic"))
        last_end = match.end()

    remaining = content[last_end:]
    if remaining.strip():
        segments.extend(_paragraph_segments(remaining))

    return [
        {
            "id": f"{session_key}_{i:04d}",
            "index": i,
            "content": chunk_content,
            "chunk_type": chunk_type,
            "status": "pending",
            "skip_reason": None,
        }
        for i, (chunk_content, chunk_type) in enumerate(segments)
    ]


def _paragraph_segments(text: str) -> list[tuple[str, str]]:
    normalized = text.replace("\f", "\n\n")
    parts = re.split(r"\n[ \t]*\n", normalized)
    segments: list[tuple[str, str]] = []
    for para in parts:
        stripped = para.strip()
        if not stripped:
            continue
        if len(stripped) > 4000:
            for line_group in _split_long_block(stripped):
                if line_group.strip():
                    segments.append((line_group.strip(), "splittable"))
        else:
            segments.append((stripped, "splittable"))
    return segments


def _split_long_block(text: str, *, max_chars: int = 3500) -> list[str]:
    lines = text.splitlines()
    groups: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        line_len = len(line) + 1
        if current and current_len + line_len > max_chars:
            groups.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len
    if current:
        groups.append("\n".join(current))
    return groups


def session_chunk_stats(session: dict[str, Any]) -> dict[str, int | float]:
    chunks = session["chunks"]
    total = len(chunks)
    extracted = sum(1 for c in chunks if c["status"] == "extracted")
    skipped = sum(1 for c in chunks if c["status"] == "skipped")
    pending = sum(1 for c in chunks if c["status"] == "pending")
    skip_ratio = (skipped / total) if total else 0.0
    extracted_ratio = (extracted / total) if total else 0.0
    return {
        "total": total,
        "extracted": extracted,
        "skipped": skipped,
        "pending": pending,
        "skip_ratio": skip_ratio,
        "extracted_ratio": extracted_ratio,
    }


def validate_service_chunk_coverage(session: dict[str, Any], *, phase: str) -> None:
    """Enforce coverage rules on marked service chunks (answer_complete only)."""
    mode = session.get("ingest_mode", DEFAULT_INGEST_MODE)
    if mode != INGEST_MODE_ANSWER_COMPLETE:
        return

    stats = session_chunk_stats(session)
    total = int(stats["total"])
    if total == 0:
        return

    extracted = int(stats["extracted"])
    skipped = int(stats["skipped"])
    pending = int(stats["pending"])

    if pending:
        return  # finish_bronze_extraction handles pending explicitly

    if extracted == 0:
        raise ValueError(
            f"Cannot {phase}: answer_complete ingest requires at least one extracted service chunk. "
            "Substantive source knowledge must be written to Bronze, not skipped."
        )

    if float(stats["skip_ratio"]) > MAX_SKIP_RATIO_ANSWER_COMPLETE:
        raise ValueError(
            f"Cannot {phase}: too many service chunks skipped "
            f"({skipped}/{total} = {stats['skip_ratio']:.0%}). "
            f"Max skip ratio is {MAX_SKIP_RATIO_ANSWER_COMPLETE:.0%} in answer_complete mode."
        )

    if float(stats["extracted_ratio"]) < MIN_EXTRACTED_RATIO_ANSWER_COMPLETE:
        raise ValueError(
            f"Cannot {phase}: too few service chunks extracted "
            f"({extracted}/{total} = {stats['extracted_ratio']:.0%}). "
            f"Min extracted ratio is {MIN_EXTRACTED_RATIO_ANSWER_COMPLETE:.0%} in answer_complete mode."
        )


def _active_chunks(store: StorageProvider, path: str) -> list[Chunk]:
    get_chunks = getattr(store, "get_chunks_by_path", None)
    if not callable(get_chunks):
        return []
    return [c for c in get_chunks(path) if c.status == "active"]


def _find_inventory(chunks: list[Chunk]) -> Chunk | None:
    for chunk in chunks:
        if chunk.layer != "bronze" or chunk.status != "active":
            continue
        if chunk.content.strip().startswith(INVENTORY_PREFIX):
            return chunk
    return None


def _find_artifact(chunks: list[Chunk]) -> Chunk | None:
    for chunk in chunks:
        if chunk.layer != "bronze" or chunk.status != "active":
            continue
        if chunk.content_type == "artifact" or chunk.immutable:
            if "content_sha256" in chunk.content or "sha256" in chunk.content.lower():
                return chunk
        if chunk.content_type == "artifact":
            return chunk
    return None


def _bronze_fact_count(chunks: list[Chunk], *, inventory_id: str | None, artifact_id: str | None) -> int:
    count = 0
    for chunk in chunks:
        if chunk.layer != "bronze" or chunk.status != "active":
            continue
        if inventory_id and chunk.id == inventory_id:
            continue
        if artifact_id and chunk.id == artifact_id:
            continue
        if chunk.content.strip().startswith(INVENTORY_PREFIX):
            continue
        count += 1
    return count


def validate_namespace_ready_for_complete(
    store: StorageProvider,
    session: dict[str, Any],
) -> dict[str, Any]:
    """Verify Bronze/Silver written in storage match answer_complete requirements."""
    mode = session.get("ingest_mode", DEFAULT_INGEST_MODE)
    path = session["source_namespace"]
    chunks = _active_chunks(store, path)

    artifact = _find_artifact(chunks)
    inventory = _find_inventory(chunks)
    silver = next((c for c in chunks if c.layer == "silver" and c.status == "active"), None)
    facts = _bronze_fact_count(
        chunks,
        inventory_id=inventory.id if inventory else None,
        artifact_id=artifact.id if artifact else None,
    )

    if mode != INGEST_MODE_ANSWER_COMPLETE:
        return {
            "artifact_id": artifact.id if artifact else None,
            "inventory_id": inventory.id if inventory else None,
            "silver_id": silver.id if silver else None,
            "bronze_fact_count": facts,
            "mode": mode,
        }

    errors: list[str] = []
    if artifact is None:
        errors.append(
            "missing active Bronze artifact (layer=bronze, content_type=artifact, immutable=true) "
            f"at {path}"
        )

    if inventory is None:
        errors.append(
            f"missing Bronze inventory chunk at {path} — content must start with {INVENTORY_PREFIX!r} "
            "listing every answer-critical entity (countries, fields, rules, codes, …)."
        )
    if facts < MIN_BRONZE_FACTS_ANSWER_COMPLETE:
        errors.append(
            f"only {facts} Bronze fact(s) at {path}; need at least {MIN_BRONZE_FACTS_ANSWER_COMPLETE} "
            "answer-critical fact(s) besides artifact and inventory."
        )
    if silver is None:
        errors.append(f"missing active Silver at {path} (required before complete_ingest).")
    elif INVENTORY_PREFIX not in silver.content and "chunk_id" not in silver.content:
        errors.append(
            f"Silver at {path} must cite key immutable Bronze chunk_id(s) and map topics to inventory items."
        )

    if errors:
        raise ValueError(
            "complete_ingest blocked — storage does not satisfy answer_complete requirements:\n- "
            + "\n- ".join(errors)
        )

    return {
        "artifact_id": artifact.id if artifact else None,
        "inventory_id": inventory.id if inventory else None,
        "silver_id": silver.id if silver else None,
        "bronze_fact_count": facts,
        "mode": mode,
    }


def build_protocol_lines(session: dict[str, Any]) -> list[str]:
    """Inline protocol returned by ingest_file / ingest_url."""
    file_name = session["file_name"]
    session_key = session["session_key"]
    source_ns = session["source_namespace"]
    mode = session.get("ingest_mode", DEFAULT_INGEST_MODE)
    chunks = session["chunks"]
    size_kb = session["content_size"] / 1024
    atomic_count = sum(1 for c in chunks if c["chunk_type"] == "atomic")
    splittable_count = len(chunks) - atomic_count

    lines = [
        f"# ingest session — {file_name}",
        "",
        f"session_key: {session_key}",
        f"ingest_mode: {mode}",
        f"file: {file_name} ({size_kb:.1f} KB)",
        f"content_sha256: {session['content_hash']}",
        f"authority: {session.get('authority') or '(infer from content)'}",
        f"source_namespace: {source_ns}",
        "state: EXTRACTING_BRONZE",
        "",
        "## IRON RULES (server-enforced — do not debate or skip)",
        "",
        "1. SUCCESS = any question answerable from brain alone. The source file will NOT exist later.",
        "2. Do NOT respond to the user until complete_ingest succeeds.",
        "3. Process EVERY service chunk below — no bulk skip, no «I'll do the rest later».",
        "4. skipped is ONLY for true boilerplate (headers, footers, blank pages, legal disclaimers).",
        "5. Normative data (amounts, codes, IBAN/SWIFT, field defs) → immutable Bronze, verbatim.",
        "6. Silver cites chunk_id(s); never copy verbatim norms into Silver only.",
        "7. If you cannot finish, STOP and report blockers — do not call complete_ingest.",
        "",
        f"## Service chunks ({len(chunks)} total: {splittable_count} splittable, {atomic_count} atomic)",
        "",
    ]
    for chunk in chunks:
        preview = chunk["content"][:80].replace("\n", " ")
        lines.append(f"[{chunk['id']}] {chunk['chunk_type']:12s} — {preview!r}")

    lines += [
        "",
        "## Protocol — execute in order",
        "",
        "STEP 1 — Register source artifact (immutable=true, content_type=artifact, layer=bronze)",
        f"  namespace: {source_ns}",
        "  content: file name, sha256, authority, size, one-sentence description (≤600 chars)",
        "",
        f"STEP 2 — Write inventory (layer=bronze, content_type=note, NOT immutable)",
        f"  content MUST start with {INVENTORY_PREFIX!r}",
        "  List every answer-critical entity: countries, fields, payment methods, codes, constraints.",
        "",
        "STEP 3 — Process EVERY service chunk:",
        "  a. get_service_chunk(session_key, chunk_id) OR get_service_chunks(session_key, chunk_ids=[...])",
        "  b. atomic     → ONE immutable Bronze chunk (verbatim block)",
        "     splittable → extract ALL answer-critical facts (≤600 chars each; batch_append max 10)",
        "     reference  → immutable Bronze for tables, schemas, routing numbers, legal norms",
        "     boilerplate only → mark_service_chunk(..., status='skipped', skip_reason='12+ chars')",
        "  c. After Bronze written → mark_service_chunk(..., status='extracted')",
        "",
        "STEP 4 — finish_bronze_extraction(session_key)",
        "  Server rejects if: pending chunks, >25% skipped, <50% extracted, zero extracted.",
        "",
        f"STEP 5 — list_chunks(path='{source_ns}', layer='bronze'); write Silver from Bronze ONLY",
        "  Silver maps topics → chunk_id(s). Required before complete_ingest.",
        "",
        "STEP 6 — complete_ingest(session_key)",
        "  Server verifies: artifact + inventory + facts + Silver present in storage.",
        "",
        "Do NOT respond to the user until step 6 succeeds.",
    ]
    if mode == INGEST_MODE_ROUTING:
        lines.insert(
            lines.index("## IRON RULES (server-enforced — do not debate or skip)") + 2,
            "(routing mode: lighter coverage rules — use only when user explicitly requested routing ingest)",
        )
    return lines
