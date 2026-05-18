"""Ingest session protocol, splitting, and server-side enforcement."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vertical_brain.core.models import Chunk
    from vertical_brain.storage.protocol import StorageProvider

INGEST_MODE_ANSWER_COMPLETE = "answer_complete"
INGEST_MODE_ROUTING = "routing"
INGEST_MODE_AUDIT = "audit"
DEFAULT_INGEST_MODE = INGEST_MODE_ANSWER_COMPLETE

# answer_complete: document must be answerable without the source file
MAX_SKIP_RATIO_ANSWER_COMPLETE = 0.25
MIN_EXTRACTED_RATIO_ANSWER_COMPLETE = 0.50
MIN_BRONZE_FACTS_ANSWER_COMPLETE = 1
MIN_INVENTORY_PROBE_COUNT = 3
MIN_INVENTORY_PROBE_RATIO = 0.30
MIN_INVENTORY_COVERAGE = 0.90   # 90% of inventory items must have probe citation

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
    if value not in (INGEST_MODE_ANSWER_COMPLETE, INGEST_MODE_ROUTING, INGEST_MODE_AUDIT):
        raise ValueError(
            f"ingest mode must be {INGEST_MODE_ANSWER_COMPLETE!r}, {INGEST_MODE_ROUTING!r}, "
            f"or {INGEST_MODE_AUDIT!r}, got {mode!r}"
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


_HEADING_RE = re.compile(
    r"#{1,4}\s+\S"              # markdown: # Title
    r"|\d+(\.\d+)*\.?\s+[A-Z]"  # numbered: 1.2 Title
    r"|[A-Z][A-Z\s\-&/()\d]{4,}[A-Z]$"  # ALL CAPS: SECTION TITLE
)


def _is_section_header(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or len(lines) > 3:
        return False
    first = lines[0].strip()
    return len(first) <= 120 and bool(_HEADING_RE.match(first))


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
            chunk_type = "section_header" if _is_section_header(stripped) else "splittable"
            segments.append((stripped, chunk_type))
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


def _active_chunks(store: StorageProvider, path: str, include_children: bool = False) -> list[Chunk]:
    get_chunks = getattr(store, "get_chunks_by_path", None)
    if not callable(get_chunks):
        return []
    try:
        raw = get_chunks(path, include_children=include_children)
    except TypeError:
        raw = get_chunks(path)
    return [c for c in raw if c.status == "active"]


def normalize_inventory_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


def parse_inventory_items(inventory_content: str) -> list[str]:
    """Parse bullet/numbered lines from an [INVENTORY] Bronze chunk."""
    items: list[str] = []
    for raw in inventory_content.splitlines():
        line = raw.strip()
        if not line or line.startswith(INVENTORY_PREFIX):
            continue
        if line.startswith(("-", "*", "•", "·")):
            line = line[1:].strip()
        line = re.sub(r"^\d+[.)]\s+", "", line)
        if line:
            items.append(line)
    return items


def inventory_probe_requirement(item_count: int) -> int:
    if item_count <= 0:
        return 0
    ratio_based = int(item_count * MIN_INVENTORY_PROBE_RATIO + 0.999)
    return min(item_count, max(MIN_INVENTORY_PROBE_COUNT, ratio_based))


def _inventory_item_matches(probe_item: str, inventory_items: list[str]) -> bool:
    probe_norm = normalize_inventory_label(probe_item)
    if not probe_norm:
        return False
    for item in inventory_items:
        item_norm = normalize_inventory_label(item)
        if probe_norm == item_norm or probe_norm in item_norm or item_norm in probe_norm:
            return True
    return False


def validate_inventory_probes(
    session: dict[str, Any],
    store: StorageProvider,
) -> dict[str, Any]:
    """Require spot-check probes linking inventory items to Bronze chunk_ids."""
    mode = session.get("ingest_mode", DEFAULT_INGEST_MODE)
    if mode != INGEST_MODE_ANSWER_COMPLETE:
        return {"required": 0, "submitted": 0}

    path = session["source_namespace"]
    # Include sub-namespace chunks so probes can cite Bronze facts written per-section.
    chunks = _active_chunks(store, path, include_children=True)
    inventory = find_inventory_chunk(chunks)
    if inventory is None:
        raise ValueError(
            "submit_inventory_probes blocked: write [INVENTORY] Bronze before probing coverage."
        )

    inventory_items = parse_inventory_items(inventory.content)
    required = inventory_probe_requirement(len(inventory_items))
    probes: list[dict[str, Any]] = list(session.get("inventory_probes") or [])
    if len(probes) < required:
        raise ValueError(
            f"complete_ingest blocked: submit at least {required} inventory probe(s) via "
            f"submit_inventory_probes (have {len(probes)}, inventory lists {len(inventory_items)} items). "
            "Each probe must name an inventory item and cite active Bronze chunk_id(s) that answer it."
        )

    active_ids = {c.id for c in chunks}
    errors: list[str] = []
    seen_items: set[str] = set()
    for probe in probes:
        item = str(probe.get("item", "")).strip()
        chunk_ids = probe.get("chunk_ids") or []
        if not item:
            errors.append("probe missing item label")
            continue
        if not _inventory_item_matches(item, inventory_items):
            errors.append(f"probe item {item!r} does not match any [INVENTORY] line")
            continue
        item_key = normalize_inventory_label(item)
        if item_key in seen_items:
            errors.append(f"duplicate probe for inventory item {item!r}")
            continue
        seen_items.add(item_key)
        if not chunk_ids:
            errors.append(f"probe {item!r} must cite at least one chunk_id")
            continue
        missing = [cid for cid in chunk_ids if cid not in active_ids]
        if missing:
            errors.append(f"probe {item!r} cites unknown or stale chunk_id(s): {missing}")
            continue
        bronze_ids = {
            c.id
            for c in chunks
            if c.id in chunk_ids and c.layer == "bronze" and c.status == "active"
        }
        if not bronze_ids:
            errors.append(f"probe {item!r} must cite at least one active Bronze chunk_id")

    if errors:
        raise ValueError(
            "complete_ingest blocked — inventory probe validation failed:\n- " + "\n- ".join(errors)
        )

    return {
        "required": required,
        "submitted": len(probes),
        "inventory_item_count": len(inventory_items),
    }


def calculate_coverage_score(
    session: dict[str, Any],
    store: StorageProvider,
) -> dict[str, Any]:
    """Return a coverage dict summarising inventory, section, and skip-ratio health."""
    source_ns = session["source_namespace"]
    probes: list[dict[str, Any]] = list(session.get("inventory_probes") or [])

    # --- inventory coverage ---
    inventory = find_inventory_chunk(_active_chunks(store, source_ns))
    if inventory is None:
        inventory_coverage = 0.0
        inventory_items: list[str] = []
    else:
        inventory_items = parse_inventory_items(inventory.content)

    cited_items = {normalize_inventory_label(p["item"]) for p in probes}
    if inventory is not None:
        covered = sum(
            1 for item in inventory_items if normalize_inventory_label(item) in cited_items
        )
        inventory_coverage = covered / len(inventory_items) if inventory_items else 1.0
    uncovered_items = [
        item for item in inventory_items if normalize_inventory_label(item) not in cited_items
    ]

    # --- section coverage ---
    # Find a Bronze note at source_ns that describes the sub-namespace plan
    plan_chunks = _active_chunks(store, source_ns)
    planned_sub_ns: list[str] = []
    for c in plan_chunks:
        if c.layer != "bronze" or c.status != "active":
            continue
        if "→ SOURCES/" in c.content or re.search(r"→\s+\S+/", c.content):
            for line in c.content.splitlines():
                for m in re.finditer(r"→\s*(SOURCES/\S+)", line):
                    planned_sub_ns.append(m.group(1).rstrip(",;"))

    missing_sections: list[str] = []
    for sub_ns in planned_sub_ns:
        sub_chunks = _active_chunks(store, sub_ns)
        has_bronze_fact = any(
            c.layer == "bronze" and c.content_type in ("fact", "reference")
            for c in sub_chunks
        )
        has_silver = any(c.layer == "silver" for c in sub_chunks)
        if not (has_bronze_fact and has_silver):
            missing_sections.append(sub_ns)

    section_coverage = len(missing_sections) == 0

    # --- skip ratio ---
    session_chunks = session.get("chunks") or []
    total = len(session_chunks)
    skipped = sum(1 for c in session_chunks if c["status"] == "skipped")
    skip_ratio = skipped / total if total else 0.0
    skip_ratio_ok = skip_ratio <= MAX_SKIP_RATIO_ANSWER_COMPLETE

    return {
        "inventory_coverage": round(inventory_coverage, 3),
        "uncovered_items": uncovered_items,
        "section_coverage": section_coverage,
        "missing_sections": missing_sections,
        "skip_ratio": round(skip_ratio, 3),
        "skip_ratio_ok": skip_ratio_ok,
    }


def find_inventory_chunk(chunks: list[Chunk]) -> Chunk | None:
    for chunk in chunks:
        if chunk.layer != "bronze" or chunk.status != "active":
            continue
        if chunk.content.strip().startswith(INVENTORY_PREFIX):
            return chunk
    return None


def load_inventory_items(store: StorageProvider, path: str) -> list[str]:
    inventory = find_inventory_chunk(_active_chunks(store, path))
    if inventory is None:
        return []
    return parse_inventory_items(inventory.content)


_find_inventory = find_inventory_chunk  # internal alias


def _silver_has_references(content: str, source_ns: str) -> bool:
    """Silver is valid if it cites chunk_ids or sub-namespace paths."""
    return "chunk_id" in content or (source_ns + "/") in content


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
    inventory = find_inventory_chunk(chunks)
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
    elif not _silver_has_references(silver.content, path):
        errors.append(
            f"Silver at {path} must cite Bronze chunk_id(s) or sub-namespace paths "
            f"(e.g. '{path}/SectionName')."
        )

    # Coverage score gate (answer_complete only, only if probes submitted)
    probes_submitted = len(session.get("inventory_probes") or []) if session else 0
    if mode == INGEST_MODE_ANSWER_COMPLETE and probes_submitted > 0:
        cov = calculate_coverage_score(session, store)
        if cov["inventory_coverage"] < MIN_INVENTORY_COVERAGE:
            errors.append(
                f"inventory_coverage {cov['inventory_coverage']:.0%} is below the required "
                f"{MIN_INVENTORY_COVERAGE:.0%}. Uncovered items: {cov['uncovered_items']}. "
                "Add more inventory probes via submit_inventory_probes."
            )
        if not cov["section_coverage"]:
            errors.append(
                f"section_coverage incomplete — missing Bronze or Silver in: {cov['missing_sections']}. "
                "Write Bronze facts and Silver index for each planned section."
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
    if session.get("ingest_mode") == INGEST_MODE_AUDIT:
        source_ns = session.get("source_namespace", "?")
        return [
            "AUDIT MODE — checking existing namespace coverage gaps.",
            f"Namespace: {source_ns}",
            "",
            "STEP 1 — submit_inventory_probes(session_key, probes=[{item, chunk_ids}, ...])",
            "  For each [INVENTORY] item you can answer: cite the Bronze chunk_id(s).",
            "  Read existing Bronze via list_chunks or read_context first.",
            "",
            "STEP 2 — complete_ingest(session_key)",
            "  Returns coverage_score + uncovered_items + missing_sections.",
            "  No Bronze writing required. This mode does not replace answer_complete.",
        ]

    file_name = session["file_name"]
    session_key = session["session_key"]
    source_ns = session["source_namespace"]
    mode = session.get("ingest_mode", DEFAULT_INGEST_MODE)
    chunks = session["chunks"]
    size_kb = session["content_size"] / 1024
    atomic_count = sum(1 for c in chunks if c["chunk_type"] == "atomic")
    section_count = sum(1 for c in chunks if c["chunk_type"] == "section_header")
    splittable_count = len(chunks) - atomic_count - section_count

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
        "6. Silver = index of Bronze chunks: one line per chunk — \"[chunk_id] one-line summary\".",
        "7. If you cannot finish, STOP and report blockers — do not call complete_ingest.",
        "",
        f"## Service chunks ({len(chunks)} total: {splittable_count} splittable, "
        f"{atomic_count} atomic, {section_count} section_header)",
        "",
    ]
    for chunk in chunks:
        preview = chunk["content"][:80].replace("\n", " ")
        lines.append(f"[{chunk['id']}] {chunk['chunk_type']:14s} — {preview!r}")

    lines += [
        "",
        "## Protocol — execute in order",
        "",
        "STEP 1 — Register source artifact (immutable=true, content_type=artifact, layer=bronze)",
        f"  namespace: {source_ns}",
        "  content: file name, sha256, authority, size, one-sentence description (≤600 chars)",
        "",
        "STEP 2 — Write [INVENTORY] Bronze (content_type=note, NOT immutable)",
        f"  content MUST start with {INVENTORY_PREFIX!r}",
        "  Each line = ONE answer-critical capability the system MUST be able to answer.",
        "  Format: '<Topic> — <specific fact or method>'",
        "  Examples: 'US ACH payment to BofA', 'Germany EUR IBAN wire', 'Costa Rica MT103 SWIFT'",
        "  NOT 'United States' or 'payment methods' — too vague to probe.",
        "  Every item will be spot-checked via submit_inventory_probes.",
        "",
        "STEP 3 — Plan sub-namespace structure",
        "  a. Scan chunk list for section_header chunks — these mark document section boundaries.",
        "  b. Map each section to a sub-namespace slug: {source_ns}/SectionSlug/",
        "     Rules: lowercase, hyphens, ASCII only. E.g. 'SEPA Payments' → sepa-payments.",
        "  c. Write plan as Bronze note (content_type=note) at {source_ns}:",
        "     \"Sub-namespace map: Section1 → {source_ns}/section1, Section2 → {source_ns}/section2, ...\"",
        "     This note satisfies the root Bronze fact requirement.",
        "",
        "STEP 4 — Process EVERY service chunk (grouped by section):",
        "  a. get_service_chunks(session_key, chunk_ids=[...]) — fetch all chunks in section at once",
        "  b. Write Bronze VERBATIM into the section sub-namespace (batch_append, max 10 per call):",
        "     content_type='reference': normative tables, SWIFT/BIC/IBAN/routing/account numbers — set immutable=True.",
        "     content_type='fact': prose instructions, rules, process notes — immutable=True recommended.",
        "     - section_header → mark_service_chunk(extracted); no Bronze write needed",
        "     - atomic         → ONE immutable Bronze chunk (verbatim code/table block)",
        "     - splittable     → one Bronze chunk per fact (≤600 chars, verbatim or minimal edit)",
        "     - boilerplate    → mark_service_chunk(skipped, skip_reason='12+ chars')",
        "  c. batch_mark_service_chunks(session_key, marks=[{chunk_id, status}, ...])",
        "",
        "STEP 5 — finish_bronze_extraction(session_key)",
        "  Server rejects if: pending chunks, >25% skipped, <50% extracted, zero extracted.",
        "",
        "STEP 6 — Write Silver per sub-namespace (index format)",
        "  For each sub-namespace that received Bronze chunks:",
        "    a. list_chunks(path=sub_namespace, layer='bronze')",
        "    b. update_silver(path=sub_namespace, content=index)",
        "       Index format — one line per Bronze chunk:",
        "       \"[chunk_id] one-line summary of what this chunk contains\"",
        f"  Then write root Silver at {source_ns} as section index:",
        f"    \"[{source_ns}/section1] description\\n[{source_ns}/section2] description\\n...\"",
        "",
        "STEP 6b — submit_inventory_probes(session_key, probes=[{item, chunk_ids}, ...])",
        "  Spot-check ≥30% of inventory items (min 3): each probe names an inventory line and cites",
        "  active Bronze chunk_id(s) that contain the answer. Server blocks complete_ingest without this.",
        "",
        "STEP 7 — complete_ingest(session_key)",
        "  Server verifies: artifact + inventory + facts + Silver + inventory probes in storage.",
        "",
        "Do NOT respond to the user until step 7 succeeds.",
    ]
    if mode == INGEST_MODE_ROUTING:
        lines.insert(
            lines.index("## IRON RULES (server-enforced — do not debate or skip)") + 2,
            "(routing mode: lighter coverage rules — use only when user explicitly requested routing ingest)",
        )
    return lines
