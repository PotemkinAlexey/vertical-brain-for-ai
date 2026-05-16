"""LlmGoldBuilder tests — LLM-driven Gold distillation with deterministic fallback."""
from __future__ import annotations

import json


from vertical_brain.core.gold import LlmGoldBuilder
from vertical_brain.core.models import Chunk


def _silver(content: str, confidence: float = 1.0, node_path: str = "WORK/Project") -> Chunk:
    return Chunk(node_path=node_path, content=content, layer="silver", confidence=confidence)


class _FakeLlm:
    """Controllable LLM stub."""

    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, prompt: str) -> str:
        return self._response


def _valid_response(chunks: list[Chunk]) -> str:
    return json.dumps({
        "facts": [
            {
                "content": f"Distilled: {c.content}",
                "source_silver_ids": [c.id],
                "confidence": c.confidence,
            }
            for c in chunks
        ],
        "entities": ["Entity"],
        "rules": ["Rule A"],
    })


# ── Happy path ────────────────────────────────────────────────────────────────

def test_llm_gold_builder_uses_llm_response():
    chunks = [_silver("fact one"), _silver("fact two")]
    llm = _FakeLlm(_valid_response(chunks))
    doc = LlmGoldBuilder(llm).build(chunks, node_path="WORK/Project")
    assert len(doc.facts) == 2
    assert all("Distilled:" in f.content for f in doc.facts)


def test_llm_gold_builder_each_fact_has_source_silver_ids():
    chunks = [_silver("important decision")]
    llm = _FakeLlm(_valid_response(chunks))
    doc = LlmGoldBuilder(llm).build(chunks, node_path="WORK/Project")
    assert doc.facts[0].source_silver_ids == [chunks[0].id]


def test_llm_gold_builder_entities_and_rules_parsed():
    chunks = [_silver("fact")]
    llm = _FakeLlm(_valid_response(chunks))
    doc = LlmGoldBuilder(llm).build(chunks, node_path="WORK/Project")
    assert "Entity" in doc.entities
    assert "Rule A" in doc.rules


def test_llm_gold_builder_strips_markdown_code_fence():
    chunks = [_silver("fact one")]
    raw = "```json\n" + _valid_response(chunks) + "\n```"
    doc = LlmGoldBuilder(_FakeLlm(raw)).build(chunks, node_path="WORK/Project")
    assert len(doc.facts) == 1
    assert doc.facts[0].content == "Distilled: fact one"


# ── Fallback on bad LLM output ────────────────────────────────────────────────

def test_llm_gold_builder_falls_back_on_invalid_json():
    chunks = [_silver("fact one"), _silver("fact two")]
    doc = LlmGoldBuilder(_FakeLlm("this is not json")).build(chunks, node_path="WORK/Project")
    # Deterministic fallback: one fact per Silver chunk
    assert len(doc.facts) == len(chunks)
    assert doc.facts[0].content == "fact one"


def test_llm_gold_builder_falls_back_on_empty_facts_list():
    chunks = [_silver("fact one")]
    doc = LlmGoldBuilder(_FakeLlm('{"facts":[],"entities":[],"rules":[]}')).build(
        chunks, node_path="WORK/Project"
    )
    assert len(doc.facts) == 1
    assert doc.facts[0].content == "fact one"


def test_llm_gold_builder_falls_back_when_llm_raises():
    class _BrokenLlm:
        def complete(self, prompt: str) -> str:
            raise RuntimeError("connection timeout")

    chunks = [_silver("critical fact")]
    doc = LlmGoldBuilder(_BrokenLlm()).build(chunks, node_path="WORK/Project")
    assert len(doc.facts) == 1
    assert doc.facts[0].content == "critical fact"


def test_llm_gold_builder_discards_source_ids_not_in_silver():
    """Facts whose source IDs don't match any real Silver chunk are dropped; fallback fires."""
    chunks = [_silver("fact")]
    payload = json.dumps({
        "facts": [{"content": "Hallucinated", "source_silver_ids": ["nonexistent-id"], "confidence": 0.9}],
        "entities": [],
        "rules": [],
    })
    doc = LlmGoldBuilder(_FakeLlm(payload)).build(chunks, node_path="WORK/Project")
    # Hallucinated fact has no valid lineage → dropped → fallback builder fires.
    contents = [f.content for f in doc.facts]
    assert "Hallucinated" not in contents
    assert chunks[0].content in contents


# ── Empty input ────────────────────────────────────────────────────────────────

def test_llm_gold_builder_empty_silver_returns_empty_document():
    doc = LlmGoldBuilder(_FakeLlm("anything")).build([], node_path="WORK/Project")
    assert doc.facts == []
    assert doc.entities == []


# ── LlmProvider protocol compatibility ───────────────────────────────────────

def test_mock_llm_satisfies_llm_provider_protocol():
    from vertical_brain.core.gold import LlmProvider
    from vertical_brain.llm.mock_llm import MockLLM
    assert isinstance(MockLLM(), LlmProvider)
