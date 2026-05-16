"""GoldDocument / GoldBuilder tests."""
from __future__ import annotations


import pytest

from vertical_brain.core.gold import GoldBuilder, GoldDocument, GoldFact
from vertical_brain.core.models import Chunk


def _silver(content: str, confidence: float = 1.0) -> Chunk:
    return Chunk(node_path="WORK/Project", content=content, layer="silver", confidence=confidence)


# ── GoldDocument serialization ────────────────────────────────────────────────

def test_gold_document_round_trips_via_content():
    doc = GoldDocument(
        facts=[GoldFact("fact one", ["id1"], 0.9)],
        entities=["Entity"],
        rules=["Rule A"],
        node_path="WORK/Project",
        built_from_silver_ids=["id1"],
    )
    content = doc.to_gold_content()
    recovered = GoldDocument.from_gold_content(content, "WORK/Project")
    assert len(recovered.facts) == 1
    assert recovered.facts[0].content == "fact one"
    assert recovered.facts[0].source_silver_ids == ["id1"]
    assert recovered.entities == ["Entity"]
    assert recovered.rules == ["Rule A"]


def test_from_gold_content_handles_empty_string():
    doc = GoldDocument.from_gold_content("", "WORK/Project")
    assert doc.facts == []
    assert doc.entities == []


def test_from_gold_content_handles_invalid_json():
    doc = GoldDocument.from_gold_content("{not-json}", "WORK/Project")
    assert doc.facts == []


# ── GoldBuilder ───────────────────────────────────────────────────────────────

def test_gold_builder_produces_one_fact_per_silver_chunk():
    chunks = [_silver("fact alpha"), _silver("fact beta"), _silver("fact gamma")]
    doc = GoldBuilder().build(chunks, node_path="WORK/Project")
    assert len(doc.facts) == len(chunks)
    assert {f.content for f in doc.facts} == {"fact alpha", "fact beta", "fact gamma"}


def test_gold_builder_each_fact_references_source_chunk():
    c = _silver("important decision")
    doc = GoldBuilder().build([c], node_path="WORK/Project")
    assert doc.facts[0].source_silver_ids == [c.id]


def test_gold_builder_built_from_silver_ids_matches_input():
    chunks = [_silver("a"), _silver("b")]
    doc = GoldBuilder().build(chunks, node_path="WORK/Project")
    assert set(doc.built_from_silver_ids) == {c.id for c in chunks}


def test_gold_builder_empty_silver_returns_empty_document():
    doc = GoldBuilder().build([], node_path="WORK/Project")
    assert doc.facts == []
    assert doc.entities == []


def test_gold_builder_confidence_propagated_from_silver():
    c = _silver("low confidence fact", confidence=0.4)
    doc = GoldBuilder().build([c], node_path="WORK/Project")
    assert doc.facts[0].confidence == pytest.approx(0.4)


def test_gold_builder_previous_gold_is_ignored_for_facts():
    """Previous Gold must not be copied into the new document."""
    previous = GoldDocument(
        facts=[GoldFact("stale fact from last run", [], 1.0)],
        node_path="WORK/Project",
    )
    new_chunk = _silver("fresh current fact")
    doc = GoldBuilder().build([new_chunk], node_path="WORK/Project", previous_gold=previous)
    contents = [f.content for f in doc.facts]
    assert "stale fact from last run" not in contents
    assert "fresh current fact" in contents


def test_distil_prompt_contains_silver_chunk_ids():
    c = _silver("critical architecture decision")
    prompt = GoldBuilder().distil_prompt([c], previous_gold=None)
    assert c.id in prompt
    assert "SOURCE SILVER CHUNKS" in prompt


def test_distil_prompt_marks_previous_gold_as_reference_only():
    prev = GoldDocument(facts=[GoldFact("old fact", [], 1.0)], node_path="WORK/Project")
    prompt = GoldBuilder().distil_prompt([_silver("new fact")], previous_gold=prev)
    assert "PREVIOUS GOLD" in prompt
    assert "reference only" in prompt.lower() or "do not copy" in prompt.lower()
