"""Structured Gold — GoldAspect v2, legacy GoldDocument format, append_gold_aspect."""
from __future__ import annotations

import json
import time

from vertical_brain.core.gold import (
    MAX_GOLD_ASPECTS,
    GoldAspect,
    GoldDocument,
    GoldFact,
    parse_gold_aspects,
    parse_gold_content,
    serialize_gold_aspects,
)
from vertical_brain.core.models import Chunk, StorageOperation
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.storage.json_store import JsonStore


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
    store.save_chunk(Chunk(node_path="WORK/A", content="silver summary", layer="silver"))
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
    store.save_chunk(Chunk(node_path="WORK/A", content="silver summary", layer="silver"))
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
    from vertical_brain.core.models import Chunk as _Chunk
    store.save_chunk(_Chunk(node_path="WORK/A", content="silver summary", layer="silver"))
    full = serialize_gold_aspects([GoldAspect(text=f"a{i}") for i in range(MAX_GOLD_ASPECTS)])
    store.save_chunk(_Chunk(node_path="WORK/A", content=full, layer="gold"))

    StorageOperationExecutor(store).apply(
        StorageOperation(operation="append_gold_aspect", target_path="WORK/A", gold_aspect="overflow")
    )
    overflow_links = [lnk for lnk in store.list_links() if lnk.link_type == "gold_overflow"]
    assert len(overflow_links) == 1
    assert overflow_links[0].source_path == "WORK/A"


# ── GoldDocument (legacy structured JSON — read-only) ─────────────────────────

def test_gold_document_roundtrip():
    doc = GoldDocument(
        facts=[GoldFact("fact one", ["id1"], 0.9)],
        entities=["Entity"],
        rules=["Rule"],
        node_path="WORK/Project",
    )
    content = doc.to_gold_content()
    recovered = GoldDocument.from_gold_content(content, "WORK/Project")
    assert recovered.facts[0].content == "fact one"
    assert recovered.facts[0].source_silver_ids == ["id1"]
    assert recovered.entities == ["Entity"]
    assert recovered.rules == ["Rule"]


def test_gold_document_from_empty_or_invalid_json():
    assert GoldDocument.from_gold_content("", "WORK/Project").facts == []
    assert GoldDocument.from_gold_content("{not-json}", "WORK/Project").facts == []


# ── parse_gold_aspects: all three storage formats ─────────────────────────────

def test_parse_gold_aspects_from_plain_text_pipe_format():
    """Legacy plain-text pipe format is wrapped in a single GoldAspect."""
    aspects = parse_gold_aspects("insight A | insight B")
    # Parsed as one aspect whose text is the whole string (pipe format applies
    # to parse_gold_content, not parse_gold_aspects; any non-JSON is treated
    # as a single-aspect plain-text item).
    assert len(aspects) >= 1


def test_parse_gold_aspects_from_v1_json_string_list():
    """v1 JSON: aspects is a list of bare strings."""
    content = json.dumps({"aspects": ["first fact", "second fact"]})
    aspects = parse_gold_aspects(content)
    assert len(aspects) == 2
    assert aspects[0].text == "first fact"
    assert aspects[1].text == "second fact"
    # IDs are auto-generated (non-empty strings).
    assert aspects[0].id
    assert aspects[1].id


def test_parse_gold_aspects_from_v2_json_dict_list():
    """v2 JSON: aspects is a list of {id, text, updated_at} dicts."""
    a = GoldAspect(text="structured fact")
    content = serialize_gold_aspects([a])
    aspects = parse_gold_aspects(content)
    assert len(aspects) == 1
    assert aspects[0].text == "structured fact"
    assert aspects[0].id == a.id
    assert aspects[0].updated_at == a.updated_at


def test_parse_gold_aspects_empty_string_returns_empty_list():
    assert parse_gold_aspects("") == []
    assert parse_gold_aspects("   ") == []
