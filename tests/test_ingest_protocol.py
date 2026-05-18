"""Tests for ingest protocol enforcement."""
from __future__ import annotations

import json
import pytest

from vertical_brain.core.models import Chunk
from vertical_brain.mcp.ingest_protocol import (
    INGEST_MODE_ANSWER_COMPLETE,
    INVENTORY_PREFIX,
    build_protocol_lines,
    inventory_probe_requirement,
    parse_inventory_items,
    session_chunk_stats,
    split_service_chunks,
    validate_inventory_probes,
    validate_namespace_ready_for_complete,
    validate_service_chunk_coverage,
    validate_skip_reason,
)
from vertical_brain.storage.sqlite_store import SQLiteStore


def test_split_service_chunks_handles_form_feed():
    content = "Page one line\n\n\f\nPage two with data\n\nMore on page two"
    chunks = split_service_chunks(content, "sess")
    assert len(chunks) >= 2
    joined = "\n".join(c["content"] for c in chunks)
    assert "Page one" in joined
    assert "Page two" in joined


def test_validate_skip_reason_rejects_weak_reason():
    with pytest.raises(ValueError, match="at least 12 characters"):
        validate_skip_reason("test", mode=INGEST_MODE_ANSWER_COMPLETE)


def test_validate_skip_reason_rejects_bulk_close():
    with pytest.raises(ValueError, match="bulk-closing"):
        validate_skip_reason("bulk skip all remaining chunks to finish", mode=INGEST_MODE_ANSWER_COMPLETE)


def test_answer_complete_rejects_all_skipped():
    session = {
        "ingest_mode": INGEST_MODE_ANSWER_COMPLETE,
        "chunks": [
            {"status": "skipped"},
            {"status": "skipped"},
            {"status": "skipped"},
        ],
    }
    with pytest.raises(ValueError, match="at least one extracted"):
        validate_service_chunk_coverage(session, phase="finish_bronze_extraction")


def test_answer_complete_rejects_high_skip_ratio():
    session = {
        "ingest_mode": INGEST_MODE_ANSWER_COMPLETE,
        "chunks": [
            {"status": "extracted"},
            {"status": "skipped"},
            {"status": "skipped"},
            {"status": "skipped"},
        ],
    }
    with pytest.raises(ValueError, match="too many service chunks skipped"):
        validate_service_chunk_coverage(session, phase="finish_bronze_extraction")


def test_complete_ingest_requires_inventory_and_facts(tmp_path):
    store = SQLiteStore(root=tmp_path)
    path = "SOURCES/doc"
    store.save_chunk(
        Chunk(
            node_path=path,
            layer="bronze",
            content_type="artifact",
            immutable=True,
            content="doc.pdf | content_sha256: abc",
        )
    )
    session = {
        "ingest_mode": INGEST_MODE_ANSWER_COMPLETE,
        "source_namespace": path,
        "chunks": [{"status": "extracted"}],
    }
    with pytest.raises(ValueError, match="inventory"):
        validate_namespace_ready_for_complete(store, session)

    store.save_chunk(
        Chunk(
            node_path=path,
            layer="bronze",
            content_type="note",
            content=f"{INVENTORY_PREFIX} US wire, UK wire",
        )
    )
    with pytest.raises(ValueError, match="Bronze fact"):
        validate_namespace_ready_for_complete(store, session)

    store.save_chunk(
        Chunk(node_path=path, layer="bronze", content_type="fact", content="US wire ABA 026009593")
    )
    with pytest.raises(ValueError, match="Silver"):
        validate_namespace_ready_for_complete(store, session)

    store.save_chunk(
        Chunk(
            node_path=path,
            layer="silver",
            content_type="note",
            content=f"US wire (sources: chunk_id=abc)",
        )
    )
    result = validate_namespace_ready_for_complete(store, session)
    assert result["bronze_fact_count"] == 1


def test_parse_inventory_items():
    content = f"{INVENTORY_PREFIX}\n- US wire\n- UK wire\n1. SEPA credit"
    items = parse_inventory_items(content)
    assert items == ["US wire", "UK wire", "SEPA credit"]


def test_inventory_probe_requirement():
    assert inventory_probe_requirement(10) == 3
    assert inventory_probe_requirement(20) == 6


def test_validate_inventory_probes_requires_minimum(tmp_path):
    store = SQLiteStore(root=tmp_path)
    path = "SOURCES/doc"
    fact = Chunk(node_path=path, layer="bronze", content="US wire routing 026009593")
    store.save_chunk(fact)
    store.save_chunk(
        Chunk(
            node_path=path,
            layer="bronze",
            content_type="note",
            content=f"{INVENTORY_PREFIX}\n- US wire\n- UK wire\n- SEPA\n- CHAPS\n- ACH",
        )
    )
    session = {
        "ingest_mode": INGEST_MODE_ANSWER_COMPLETE,
        "source_namespace": path,
        "inventory_probes": [{"item": "US wire", "chunk_ids": [fact.id]}],
    }
    with pytest.raises(ValueError, match="at least 3 inventory probe"):
        validate_inventory_probes(session, store)


def test_build_protocol_lines_includes_iron_rules():
    session = {
        "session_key": "abc",
        "file_name": "doc.pdf",
        "authority": "bofa",
        "source_namespace": "SOURCES/doc",
        "content_hash": "hash",
        "content_size": 1024,
        "ingest_mode": INGEST_MODE_ANSWER_COMPLETE,
        "chunks": [{"id": "abc_0000", "chunk_type": "splittable", "content": "sample", "status": "pending"}],
    }
    text = "\n".join(build_protocol_lines(session))
    assert "IRON RULES" in text
    assert INVENTORY_PREFIX in text
    assert "complete_ingest" in text
