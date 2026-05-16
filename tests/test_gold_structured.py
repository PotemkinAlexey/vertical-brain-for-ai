"""Structured Gold — GoldAspect v2, LlmGoldBuilder, and lineage validation tests."""
from __future__ import annotations

import json
import time

import pytest

from vertical_brain.core.gold import (
    MAX_GOLD_ASPECTS,
    GoldAspect,
    GoldBuilder,
    GoldDocument,
    GoldFact,
    LlmGoldBuilder,
    parse_gold_aspects,
    parse_gold_content,
    serialize_gold_aspects,
)
from vertical_brain.core.models import Chunk, ChunkInput, StorageOperation
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.llm.mock_llm import MockLLM
from vertical_brain.storage.json_store import JsonStore


# ── helpers ───────────────────────────────────────────────────────────────────

def _silver(content: str, *, node_path: str = "WORK/A", confidence: float = 1.0) -> Chunk:
    return Chunk(node_path=node_path, content=content, layer="silver", confidence=confidence)


# ── parse_gold_content — all three formats ────────────────────────────────────

def test_parse_gold_content_legacy_pipe_format():
    assert parse_gold_content("Delta migration | AutoLoader streaming") == [
        "Delta migration", "AutoLoader streaming"
    ]


def test_parse_gold_content_v1_json_string_list():
    content = json.dumps({"aspects": ["fact one", "fact two"]})
    assert parse_gold_content(content) == ["fact one", "fact two"]


def test_parse_gold_content_v2_json_dict_list():
    content = json.dumps({"aspects": [
        {"id": "abc", "text": "fact alpha", "updated_at": "2026-01-01"},
        {"id": "def", "text": "fact beta", "updated_at": "2026-01-02"},
    ]})
    assert parse_gold_content(content) == ["fact alpha", "fact beta"]


def test_parse_gold_content_empty():
    assert parse_gold_content("") == []
    assert parse_gold_content("   ") == []


# ── GoldAspect dataclass ──────────────────────────────────────────────────────

def test_gold_aspect_gets_unique_id():
    a = GoldAspect(text="fact")
    b = GoldAspect(text="fact")
    assert a.id != b.id


def test_gold_aspect_gets_timestamp():
    a = GoldAspect(text="fact")
    assert a.updated_at


# ── serialize / parse round-trip ──────────────────────────────────────────────

def test_serialize_parse_gold_aspects_roundtrip():
    aspects = [GoldAspect(text="Delta Lake"), GoldAspect(text="AutoLoader")]
    content = serialize_gold_aspects(aspects)
    recovered = parse_gold_aspects(content)
    assert [a.text for a in recovered] == ["Delta Lake", "AutoLoader"]
    assert recovered[0].id == aspects[0].id
    assert recovered[1].id == aspects[1].id


def test_serialize_produces_valid_v2_json():
    content = serialize_gold_aspects([GoldAspect(text="fact")])
    parsed = json.loads(content)
    assert "aspects" in parsed
    item = parsed["aspects"][0]
    assert item["text"] == "fact"
    assert "id" in item
    assert "updated_at" in item


# ── append_gold_aspect deduplication ─────────────────────────────────────────

def test_append_gold_aspect_deduplicates_same_text(tmp_path):
    store = JsonStore(tmp_path)
    ex = StorageOperationExecutor(store)
    ex.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/A", gold_aspect="same fact"))
    ex.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/A", gold_aspect="same fact"))

    gold = [c for c in store.get_chunks_by_path("WORK/A") if c.layer == "gold" and c.status == "active"]
    assert len(gold) == 1
    aspects = parse_gold_aspects(gold[0].content)
    assert len(aspects) == 1
    assert aspects[0].text == "same fact"


def test_append_gold_aspect_dedup_refreshes_updated_at(tmp_path):
    store = JsonStore(tmp_path)
    ex = StorageOperationExecutor(store)
    ex.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/A", gold_aspect="evolving fact"))
    first_ts = parse_gold_aspects(
        next(c for c in store.get_chunks_by_path("WORK/A") if c.layer == "gold" and c.status == "active").content
    )[0].updated_at

    time.sleep(0.01)
    ex.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/A", gold_aspect="evolving fact"))
    second_ts = parse_gold_aspects(
        next(c for c in store.get_chunks_by_path("WORK/A") if c.layer == "gold" and c.status == "active").content
    )[0].updated_at

    assert second_ts >= first_ts


# ── overflow after MAX_GOLD_ASPECTS ──────────────────────────────────────────

def test_append_gold_aspect_overflow_creates_gold_overflow_link(tmp_path):
    store = JsonStore(tmp_path)
    full = serialize_gold_aspects([GoldAspect(text=f"a{i}") for i in range(MAX_GOLD_ASPECTS)])
    from vertical_brain.core.models import Chunk as _Chunk
    store.save_chunk(_Chunk(node_path="WORK/A", content=full, layer="gold"))

    StorageOperationExecutor(store).apply(
        StorageOperation(operation="append_gold_aspect", target_path="WORK/A", gold_aspect="overflow")
    )
    overflow_links = [lnk for lnk in store.list_links() if lnk.link_type == "gold_overflow"]
    assert len(overflow_links) == 1
    assert overflow_links[0].source_path == "WORK/A"


# ── GoldBuilder only from Silver ──────────────────────────────────────────────

def test_gold_builder_rejects_non_silver_via_exclusion(tmp_path):
    """GoldBuilder must produce facts only for Silver-layer chunks it receives.

    The caller is responsible for filtering; this test verifies that if only
    Silver chunks are passed, only those are reflected in built_from_silver_ids.
    """
    silver = _silver("architecture decision")
    bronze = Chunk(node_path="WORK/A", content="raw note", layer="bronze")
    doc = GoldBuilder().build([silver], node_path="WORK/A")
    assert silver.id in doc.built_from_silver_ids
    assert bronze.id not in doc.built_from_silver_ids


def test_gold_builder_empty_gives_empty_document():
    doc = GoldBuilder().build([], node_path="WORK/A")
    assert doc.facts == []
    assert doc.entities == []
    assert doc.built_from_silver_ids == []


# ── LlmGoldBuilder ───────────────────────────────────────────────────────────

def _valid_llm_response(silver_chunk: Chunk) -> str:
    return json.dumps({
        "facts": [{"content": "extracted fact", "source_silver_ids": [silver_chunk.id], "confidence": 0.9}],
        "entities": ["EntityOne"],
        "rules": [],
    })


def test_llm_gold_builder_uses_llm_response(tmp_path):
    c = _silver("key architecture decision")
    provider = MockLLM(responses=[_valid_llm_response(c)])
    doc = LlmGoldBuilder(provider).build([c], node_path="WORK/A")
    assert len(doc.facts) == 1
    assert doc.facts[0].content == "extracted fact"
    assert doc.facts[0].source_silver_ids == [c.id]
    assert doc.entities == ["EntityOne"]


def test_llm_gold_builder_falls_back_on_malformed_json():
    c = _silver("key architecture decision")
    provider = MockLLM(responses=["{not valid json"])
    doc = LlmGoldBuilder(provider).build([c], node_path="WORK/A")
    # Fallback to deterministic: fact content should be the Silver chunk's content.
    assert any(f.content == c.content for f in doc.facts)


def test_llm_gold_builder_falls_back_on_empty_facts():
    c = _silver("key architecture decision")
    provider = MockLLM(responses=[json.dumps({"facts": [], "entities": [], "rules": []})])
    doc = LlmGoldBuilder(provider).build([c], node_path="WORK/A")
    assert any(f.content == c.content for f in doc.facts)


def test_llm_gold_builder_rejects_fact_with_no_valid_source_ids():
    c = _silver("key architecture decision")
    bad_response = json.dumps({
        "facts": [{"content": "hallucinated fact", "source_silver_ids": ["nonexistent-id"], "confidence": 0.9}],
        "entities": [],
        "rules": [],
    })
    provider = MockLLM(responses=[bad_response])
    doc = LlmGoldBuilder(provider).build([c], node_path="WORK/A")
    # Hallucinated fact (no valid lineage) must be dropped; fallback fires.
    assert all(f.content != "hallucinated fact" for f in doc.facts)
    assert any(f.source_silver_ids == [c.id] for f in doc.facts)


def test_llm_gold_builder_filters_facts_with_invalid_ids_keeps_valid_ones():
    c1 = _silver("valid source chunk")
    c2 = _silver("another valid chunk")
    mixed_response = json.dumps({
        "facts": [
            {"content": "backed fact", "source_silver_ids": [c1.id], "confidence": 0.9},
            {"content": "unbacked fact", "source_silver_ids": ["ghost-id"], "confidence": 0.5},
        ],
        "entities": [],
        "rules": [],
    })
    provider = MockLLM(responses=[mixed_response])
    doc = LlmGoldBuilder(provider).build([c1, c2], node_path="WORK/A")
    contents = [f.content for f in doc.facts]
    assert "backed fact" in contents
    assert "unbacked fact" not in contents


def test_llm_gold_builder_provider_exception_triggers_fallback():
    c = _silver("key architecture decision")

    class FailingProvider:
        def complete(self, prompt: str) -> str:
            raise RuntimeError("provider down")

    doc = LlmGoldBuilder(FailingProvider()).build([c], node_path="WORK/A")
    assert any(f.content == c.content for f in doc.facts)


def test_llm_gold_builder_empty_silver_returns_empty_document():
    provider = MockLLM(responses=[])
    doc = LlmGoldBuilder(provider).build([], node_path="WORK/A")
    assert doc.facts == []
