from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from uuid import uuid4

MAX_GOLD_ASPECTS = 20
GOLD_ASPECT_EMBED_PREFIX = "gold-aspect:v1:"


def normalize_gold_aspect_text(text: str) -> str:
    """Return canonical text for Gold aspect vector cache keys."""
    return " ".join(text.split())


def gold_aspect_embed_key(text: str) -> str:
    """Return persistent vector-cache key for one Gold aspect text."""
    normalized = normalize_gold_aspect_text(text)
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return f"{GOLD_ASPECT_EMBED_PREFIX}{digest}"


def parse_gold_content(content: str) -> list[str]:
    """Parse Gold chunk content into a list of aspect text strings.

    Supports all three formats for backward compatibility:
    - Plain text: ``"Delta migration | AutoLoader streaming"``
    - v1 JSON: ``{"aspects": ["Delta migration", ...]}``
    - v2 JSON: ``{"aspects": [{"id": "...", "text": "...", "updated_at": "..."}]}``
    - Structured GoldDocument JSON: ``{"facts": [{"content": "..."}]}``
    """
    structured_facts = _parse_structured_gold_facts(content)
    if structured_facts is not None:
        return structured_facts
    return [a.text for a in parse_gold_aspects(content)]


def _parse_structured_gold_facts(content: str) -> list[str] | None:
    text = content.strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("facts"), list):
        return None
    facts: list[str] = []
    for item in parsed["facts"]:
        if not isinstance(item, dict):
            continue
        fact = item.get("content")
        if isinstance(fact, str) and fact.strip():
            facts.append(fact.strip())
    return facts


# ── Structured Gold aspects (v2) ──────────────────────────────────────────────

def _utc_now() -> str:
    from vertical_brain.core.models import utc_now
    return utc_now()


@dataclass
class GoldAspect:
    """A single Gold aspect with stable identity for dedup and refresh tracking."""
    text: str
    id: str = field(default_factory=lambda: str(uuid4()))
    updated_at: str = field(default_factory=_utc_now)


def parse_gold_aspects(content: str) -> list[GoldAspect]:
    """Parse Gold chunk content into a list of GoldAspect objects.

    Handles all three storage formats:
    - Plain text: ``"Delta migration | AutoLoader streaming"``
    - v1 JSON: ``{"aspects": ["Delta migration", ...]}``
    - v2 JSON: ``{"aspects": [{"id": "...", "text": "...", "updated_at": "..."}]}``
    """
    text = content.strip()
    if not text:
        return []
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("aspects"), list):
            aspects: list[GoldAspect] = []
            for item in parsed["aspects"]:
                if isinstance(item, dict):
                    t = item.get("text", "")
                    if not t or not str(t).strip():
                        continue
                    aspects.append(GoldAspect(
                        text=str(t).strip(),
                        id=item.get("id", str(uuid4())),
                        updated_at=item.get("updated_at", _utc_now()),
                    ))
                else:
                    t = str(item).strip()
                    if t:
                        aspects.append(GoldAspect(text=t))
            return aspects
    return [GoldAspect(text=part) for part in (s.strip() for s in text.split(" | ")) if part]


def gold_embed_text(content: str) -> str:
    """Return clean embedding text for a Gold chunk.

    Strips JSON structure (IDs, timestamps) and joins aspect texts with `` | ``.
    This produces a compact, noise-free string suitable for embedding models —
    no UUIDs, no ISO timestamps, just the semantic content of each aspect.

    Falls back to the raw *content* if parsing yields nothing (e.g. empty chunk).
    """
    texts = [t.strip() for t in parse_gold_content(content) if t.strip()]
    return " | ".join(texts) if texts else content


def serialize_gold_aspects(aspects: list[GoldAspect]) -> str:
    """Serialize a list of GoldAspects to the v2 JSON storage format."""
    return json.dumps(
        {"aspects": [{"id": a.id, "text": a.text, "updated_at": a.updated_at} for a in aspects]},
        ensure_ascii=False,
    )


# ── Immutable Gold reduction structures ──────────────────────────────────────

@dataclass
class GoldFact:
    """A single extracted fact with a strict lineage back to Silver sources."""
    content: str
    source_silver_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class GoldDocument:
    """Structured Gold layer document derived exclusively from Silver chunks.

    Previous Gold output may be used only as a read-only cache for delta
    computation — it must never be the sole input to a new Gold document.
    Every fact carries ``source_silver_ids`` that trace lineage back to Bronze.
    """
    facts: list[GoldFact] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    node_path: str = ""
    built_from_silver_ids: list[str] = field(default_factory=list)

    def to_gold_content(self) -> str:
        """Serialize to a Gold chunk content string (structured JSON)."""
        return json.dumps(
            {
                "facts": [
                    {
                        "content": f.content,
                        "source_silver_ids": f.source_silver_ids,
                        "confidence": f.confidence,
                    }
                    for f in self.facts
                ],
                "entities": self.entities,
                "rules": self.rules,
                "node_path": self.node_path,
                "built_from_silver_ids": self.built_from_silver_ids,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_gold_content(cls, content: str, node_path: str = "") -> "GoldDocument":
        """Deserialize from a Gold chunk content string."""
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return cls(node_path=node_path)
        if not isinstance(parsed, dict):
            return cls(node_path=node_path)
        facts = [
            GoldFact(
                content=f.get("content", ""),
                source_silver_ids=f.get("source_silver_ids", []),
                confidence=f.get("confidence", 1.0),
            )
            for f in parsed.get("facts", [])
            if isinstance(f, dict)
        ]
        return cls(
            facts=facts,
            entities=parsed.get("entities", []),
            rules=parsed.get("rules", []),
            node_path=node_path or parsed.get("node_path", ""),
            built_from_silver_ids=parsed.get("built_from_silver_ids", []),
        )
