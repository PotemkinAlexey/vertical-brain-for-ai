import json

from vertical_brain.core.doctor import Doctor
from vertical_brain.core.models import Chunk, Link
from vertical_brain.storage.json_store import JsonStore


def test_doctor_clean_store_reports_no_issues(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/A")
    store.save_chunk(Chunk(node_path="WORK/A", content="clean fact"))
    assert Doctor(store).run() == []


def test_doctor_finds_orphan_links(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/A")
    # Write a link directly that references a non-existent node.
    store.links_file.write_text(
        json.dumps([
            {
                "id": "lnk1",
                "source_path": "WORK/A",
                "target_path": "WORK/GHOST",
                "link_type": "peer",
                "reason": "test",
                "created_at": "2020-01-01T00:00:00+00:00",
            }
        ]),
        encoding="utf-8",
    )
    issues = Doctor(store).run()
    assert any(i.check == "orphan_link" for i in issues)


def test_doctor_finds_chunks_missing_nodes(tmp_path):
    store = JsonStore(tmp_path)
    store.chunks_file.write_text(
        json.dumps([
            {
                "node_path": "WORK/MISSING",
                "content": "orphan chunk",
                "layer": "bronze",
                "content_type": "note",
                "status": "active",
                "source": "manual",
                "confidence": 1.0,
                "lineage": [],
                "id": "c1",
                "created_at": "2020-01-01T00:00:00+00:00",
                "updated_at": "2020-01-01T00:00:00+00:00",
            }
        ]),
        encoding="utf-8",
    )
    issues = Doctor(store).run()
    assert any(i.check == "chunk_missing_node" for i in issues)


def test_doctor_finds_duplicate_active_chunks_by_hash(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/A")
    store.save_chunk(Chunk(node_path="WORK/A", content="same content"))
    store.save_chunk(Chunk(node_path="WORK/A", content="same content"))
    issues = Doctor(store).run()
    assert any(i.check == "duplicate_active_chunk" for i in issues)


def test_doctor_json_output_is_stable(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/A")
    store.save_chunk(Chunk(node_path="WORK/A", content="dup"))
    store.save_chunk(Chunk(node_path="WORK/A", content="dup"))
    issues = Doctor(store).run()
    serialized = [
        {"severity": i.severity, "check": i.check, "message": i.message, "path": i.path}
        for i in issues
    ]
    # round-trips through JSON without error
    assert json.loads(json.dumps(serialized)) == serialized
