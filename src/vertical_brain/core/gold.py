from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vertical_brain.core.models import Chunk


@runtime_checkable
class LlmProvider(Protocol):
    """Minimal LLM interface required by GoldBuilder — compatible with router.LLMProvider."""
    def complete(self, prompt: str) -> str: ...


def parse_gold_content(content: str) -> list[str]:
    """Parse Gold chunk content into a list of aspects.

    Supports both formats for backward compatibility:
    - Plain text: ``"Delta migration | AutoLoader streaming"``
    - Structured JSON: ``{"aspects": ["Delta migration", ...], "last_updated": "..."}``
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
            return [str(aspect) for aspect in parsed["aspects"] if str(aspect).strip()]
    return [aspect.strip() for aspect in text.split(" | ") if aspect.strip()]


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
    """GoldBuilder that uses an LLM to distil Silver chunks into a GoldDocument.

    The LLM is called with the structured prompt from ``distil_prompt()``.
    If the model returns malformed JSON, or if ``silver_chunks`` is empty,
    the call transparently falls back to the deterministic ``GoldBuilder.build()``.

    Usage::

        builder = LlmGoldBuilder(llm=my_llm_provider)
        doc = builder.build(silver_chunks, node_path="WORK/Project")
    """

    def __init__(self, llm: LlmProvider) -> None:
        self._llm = llm

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
            raw = self._llm.complete(prompt)
            doc = self._parse_llm_response(raw, silver_chunks, node_path)
            if doc is not None:
                return doc
        except Exception:
            pass
        return super().build(silver_chunks, node_path=node_path, previous_gold=previous_gold)

    def _parse_llm_response(
        self,
        raw: str,
        silver_chunks: list[Chunk],
        node_path: str,
    ) -> GoldDocument | None:
        """Parse LLM JSON response into GoldDocument; return None on any parse failure."""
        text = raw.strip()
        # Strip markdown code fences if the model wrapped the JSON.
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None

        silver_ids = {c.id for c in silver_chunks}
        facts_raw = payload.get("facts", [])
        if not isinstance(facts_raw, list):
            return None

        facts: list[GoldFact] = []
        for item in facts_raw:
            if not isinstance(item, dict):
                continue
            content = item.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            source_ids = [
                sid for sid in item.get("source_silver_ids", [])
                if isinstance(sid, str) and sid in silver_ids
            ]
            confidence = item.get("confidence", 1.0)
            if not isinstance(confidence, (int, float)):
                confidence = 1.0
            facts.append(GoldFact(
                content=content.strip(),
                source_silver_ids=source_ids,
                confidence=float(confidence),
            ))

        if not facts:
            return None

        entities = [str(e) for e in payload.get("entities", []) if str(e).strip()]
        rules = [str(r) for r in payload.get("rules", []) if str(r).strip()]
        return GoldDocument(
            facts=facts,
            entities=entities[:20],
            rules=rules,
            node_path=node_path,
            built_from_silver_ids=[c.id for c in silver_chunks],
        )
