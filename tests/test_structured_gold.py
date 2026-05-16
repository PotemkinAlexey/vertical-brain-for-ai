"""Structured Gold aspects — GoldAspect v2 format tests."""
from __future__ import annotations

import json

import pytest

from vertical_brain.core.gold import (
    MAX_GOLD_ASPECTS,
    GoldAspect,
    parse_gold_aspects,
    parse_gold_content,
    serialize_gold_aspects,
)
from vertical_brain.core.models import Chunk
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.core.models import StorageOperation
from vertical_brain.storage.json_store import JsonStore


# ── GoldAspect dataclass ──────────────────────────────────────────────────────

def test_gold_aspect_gets_unique_id():
    a = GoldAspect(text="fact")
    b = GoldAspect(text="fact")
    assert a.id != b.id


def test_gold_aspect_gets_timestamp():
    a = GoldAspect(text="fact")
    assert a.updated_at


# ── serialize / parse round-trip ──────────────────────────────────────────────

def test_serialize_parse_round_trip():
    aspects = [GoldAspect(text="Delta Lake"), GoldAspect(text="AutoLoader")]
    content = serialize_gold_aspects(aspects)
    recovered = parse_gold_aspects(content)
    assert [a.text for a in recovered] == ["Delta Lake", "AutoLoader"]
    assert recovered[0].id == aspects[0].id


def test_serialize_produces_valid_json():
    content = serialize_gold_aspects([GoldAspect(text="fact")])
    parsed = json.loads(content)
    assert "aspects" in parsed
    assert isinstance(parsed["aspects"][0], dict)
    assert "id" in parsed["aspects"][0]
    assert "updated_at" in parsed["aspects"][0]


def test_parse_gold_content_still_returns_strings():
    aspects = [GoldAspect(text="fact one"), GoldAspect(text="fact two")]
    content = serialize_gold_aspects(aspects)
    assert parse_gold_content(content) == ["fact one", "fact two"]


# ── Backward compatibility ────────────────────────────────────────────────────

def test_parse_gold_aspects_handles_plain_text():
    aspects = parse_gold_aspects("Delta migration | AutoLoader streaming")
    assert [a.text for a in aspects] == ["Delta migration", "AutoLoader streaming"]


def test_parse_gold_aspects_handles_v1_json_strings():
    content = json.dumps({"aspects": ["fact one", "fact two"]})
    aspects = parse_gold_aspects(content)
    assert [a.text for a in aspects] == ["fact one", "fact two"]


def test_parse_gold_aspects_handles_empty():
    assert parse_gold_aspects("") == []


def test_parse_gold_aspects_handles_json_without_aspects_key():
    # JSON object with no "aspects" key falls through to plain-text split.
    result = parse_gold_aspects('{"other": "stuff"}')
    # Should return a single aspect with the raw string as text.
    assert len(result) == 1
    assert result[0].text == '{"other": "stuff"}'


# ── append_gold_aspect uses structured format ─────────────────────────────────

def test_append_gold_aspect_stores_structured_json(tmp_path):
    store = JsonStore(tmp_path)
    StorageOperationExecutor(store).apply(
        StorageOperation(operation="append_gold_aspect", target_path="WORK/A", gold_aspect="fact one")
    )
    gold = [c for c in store.get_chunks_by_path("WORK/A") if c.layer == "gold"]
    assert len(gold) == 1
    parsed = json.loads(gold[0].content)
    assert "aspects" in parsed
    assert parsed["aspects"][0]["text"] == "fact one"
    assert "id" in parsed["aspects"][0]
    assert "updated_at" in parsed["aspects"][0]


def test_append_gold_aspect_deduplicates_same_text(tmp_path):
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/A", gold_aspect="same fact"))
    executor.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/A", gold_aspect="same fact"))

    gold = [c for c in store.get_chunks_by_path("WORK/A") if c.layer == "gold" and c.status == "active"]
    assert len(gold) == 1
    aspects = parse_gold_aspects(gold[0].content)
    assert len(aspects) == 1
    assert aspects[0].text == "same fact"


def test_append_gold_aspect_dedup_refreshes_updated_at(tmp_path):
    import time
    store = JsonStore(tmp_path)
    executor = StorageOperationExecutor(store)
    executor.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/A", gold_aspect="evolving fact"))
    first_ts = parse_gold_aspects(
        next(c for c in store.get_chunks_by_path("WORK/A") if c.layer == "gold" and c.status == "active").content
    )[0].updated_at

    time.sleep(0.01)
    executor.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/A", gold_aspect="evolving fact"))
    second_ts = parse_gold_aspects(
        next(c for c in store.get_chunks_by_path("WORK/A") if c.layer == "gold" and c.status == "active").content
    )[0].updated_at

    assert second_ts >= first_ts


def test_append_gold_aspect_overflow_at_max_aspects(tmp_path):
    store = JsonStore(tmp_path)
    full = serialize_gold_aspects([GoldAspect(text=f"a{i}") for i in range(MAX_GOLD_ASPECTS)])
    store.save_chunk(Chunk(node_path="WORK/A", content=full, layer="gold"))

    StorageOperationExecutor(store).apply(
        StorageOperation(operation="append_gold_aspect", target_path="WORK/A", gold_aspect="overflow")
    )
    overflow_links = [lnk for lnk in store.list_links() if lnk.link_type == "gold_overflow"]
    assert len(overflow_links) == 1
