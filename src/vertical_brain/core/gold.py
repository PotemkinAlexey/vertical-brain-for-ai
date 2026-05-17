from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

if TYPE_CHECKING:
    from vertical_brain.core.models import Chunk


@runtime_checkable
class LlmProvider(Protocol):
    def complete(self, prompt: str) -> str: ...


MAX_GOLD_ASPECTS = 20


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


class GoldBuilder:
    """Builds an immutable GoldDocument from Silver chunks.

    The reduction is always derived from the provided Silver chunks — the
    previous Gold is used only as a delta cache, never as the source of truth.
    """

    def build(
        self,
        silver_chunks: list[Chunk],
        *,
        node_path: str = "",
        previous_gold: GoldDocument | None = None,
    ) -> GoldDocument:
        """Build a GoldDocument from an ordered list of active Silver chunks.

        The LLM distillation prompt is constructed here but not executed
        (no live LLM dependency). Callers that want LLM-driven reduction
        should override ``_distil`` or use a subclass.
        """
        sorted_chunks = sorted(silver_chunks, key=lambda c: c.created_at)
        if not sorted_chunks:
            return GoldDocument(node_path=node_path)

        facts = [
            GoldFact(
                content=chunk.content,
                source_silver_ids=[chunk.id],
                confidence=chunk.confidence,
            )
            for chunk in sorted_chunks
        ]
        return GoldDocument(
            facts=facts,
            entities=self._extract_entities(sorted_chunks),
            rules=[],
            node_path=node_path,
            built_from_silver_ids=[c.id for c in sorted_chunks],
        )

    def distil_prompt(self, silver_chunks: list[Chunk], previous_gold: GoldDocument | None) -> str:
        """Return the LLM prompt for structured Gold distillation.

        The model must return a JSON document matching:
        {
          "facts": [{"content": "...", "source_silver_ids": [...], "confidence": 0.9}],
          "entities": ["Entity1", "Entity2"],
          "rules": ["Rule statement"]
        }
        Previous Gold is provided as context only — it must not be copied verbatim.
        """
        silver_text = "\n".join(
            f"[{c.id}] ({c.content_type}, conf={c.confidence:.2f}) {c.content}"
            for c in sorted(silver_chunks, key=lambda c: c.created_at)
        )
        prev_text = (
            "\n".join(f.content for f in previous_gold.facts)
            if previous_gold and previous_gold.facts
            else "(none)"
        )
        return (
            f"You are building a Gold knowledge summary for namespace: {silver_chunks[0].node_path if silver_chunks else '?'}.\n\n"
            f"SOURCE SILVER CHUNKS (authoritative):\n{silver_text}\n\n"
            f"PREVIOUS GOLD (reference only — do not copy verbatim):\n{prev_text}\n\n"
            "Return ONLY valid JSON: {\"facts\": [{\"content\": \"...\", "
            "\"source_silver_ids\": [\"...\"], \"confidence\": 0.9}], "
            "\"entities\": [...], \"rules\": [...]}"
        )

    def _extract_entities(self, chunks: list[Chunk]) -> list[str]:
        seen: set[str] = set()
        entities: list[str] = []
        for chunk in chunks:
            for word in chunk.content.split():
                if len(word) > 4 and word[0].isupper() and word not in seen:
                    seen.add(word)
                    entities.append(word)
        return entities[:20]


class LlmGoldBuilder(GoldBuilder):
    """GoldBuilder that calls an LLM provider for distillation.

    Falls back to the deterministic GoldBuilder when:
    - the provider raises an exception
    - the response is not valid JSON
    - no returned fact has at least one source_silver_id matching a real Silver chunk
    """

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    def build(
        self,
        silver_chunks: list[Chunk],
        *,
        node_path: str = "",
        previous_gold: GoldDocument | None = None,
    ) -> GoldDocument:
        if not silver_chunks:
            return GoldDocument(node_path=node_path)
        prompt = self.distil_prompt(silver_chunks, previous_gold)
        try:
            response = self._provider.complete(prompt)
            doc = self._parse_llm_response(response, silver_chunks, node_path)
        except Exception:
            doc = None
        if doc is None:
            return super().build(silver_chunks, node_path=node_path, previous_gold=previous_gold)
        return doc

    def _parse_llm_response(
        self,
        response: str,
        silver_chunks: list[Chunk],
        node_path: str,
    ) -> GoldDocument | None:
        """Parse and validate the LLM JSON response.

        Returns None (triggering fallback) if JSON is malformed or every fact
        lacks a valid source_silver_id that maps back to a real Silver chunk.
        """
        try:
            parsed = json.loads(_strip_json_code_fence(response))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(parsed, dict):
            return None

        valid_silver_ids = {c.id for c in silver_chunks}
        facts: list[GoldFact] = []
        for f in parsed.get("facts", []):
            if not isinstance(f, dict):
                continue
            source_ids = [
                sid for sid in f.get("source_silver_ids", [])
                if isinstance(sid, str) and sid in valid_silver_ids
            ]
            if not source_ids:
                continue
            content = f.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            facts.append(GoldFact(
                content=content.strip(),
                source_silver_ids=source_ids,
                confidence=float(f.get("confidence", 1.0)),
            ))

        if not facts:
            return None

        return GoldDocument(
            facts=facts,
            entities=parsed.get("entities", []),
            rules=parsed.get("rules", []),
            node_path=node_path,
            built_from_silver_ids=sorted(valid_silver_ids),
        )


def _strip_json_code_fence(response: str) -> str:
    text = response.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text
