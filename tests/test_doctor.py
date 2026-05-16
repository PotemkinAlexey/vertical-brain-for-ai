import json


from vertical_brain.core.doctor import Doctor
from vertical_brain.core.models import Chunk
from vertical_brain.storage.json_store import JsonStore
from vertical_brain.storage.sqlite_store import SQLiteStore


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


def test_doctor_skips_fts_check_for_json_store(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="fact"))
    issues = Doctor(store).run()
    assert not any(i.check == "fts_stale_leak" for i in issues)


def test_doctor_detects_fts_stale_leak_in_sqlite(tmp_path):
    store = SQLiteStore(tmp_path)
    chunk = store.save_chunk(Chunk(node_path="WORK/A", content="stale fact"))
    # Simulate stale leak: chunk indexed as active but status changed in DB without re-indexing.
    store.conn.execute(
        "UPDATE chunks SET status = 'stale' WHERE id = ?", (chunk.id,)
    )
    store.conn.commit()
    # Do NOT call rebuild_search_index so FTS still has the old record.
    issues = Doctor(store).run()
    stale_issues = [i for i in issues if i.check == "fts_stale_leak"]
    assert len(stale_issues) == 1
    assert chunk.id in stale_issues[0].message
    assert stale_issues[0].severity == "warning"


def test_doctor_no_fts_stale_leak_when_index_is_clean(tmp_path):
    store = SQLiteStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="active fact"))
    issues = Doctor(store).run()
    assert not any(i.check == "fts_stale_leak" for i in issues)


# ── Item 3: Doctor duplicate detection with content_hash-or-content fallback ──

def test_doctor_duplicate_uses_content_hash_or_content_fallback(tmp_path):
    """Legacy chunks with empty content_hash must still be deduplicated via content."""
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/A")
    c1 = Chunk(node_path="WORK/A", content="same legacy fact")
    c2 = Chunk(node_path="WORK/A", content="same legacy fact")
    # Simulate legacy rows with empty content_hash (as they arrive from old data).
    c1.content_hash = ""
    c2.content_hash = ""
    store.save_chunk(c1)
    store.save_chunk(c2)
    issues = Doctor(store).run()
    dup_issues = [i for i in issues if i.check == "duplicate_active_chunk"]
    assert len(dup_issues) >= 1


def test_doctor_duplicate_message_uses_dedupe_key_wording(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/A")
    store.save_chunk(Chunk(node_path="WORK/A", content="duplicate content"))
    store.save_chunk(Chunk(node_path="WORK/A", content="duplicate content"))
    issues = Doctor(store).run()
    dup = next(i for i in issues if i.check == "duplicate_active_chunk")
    assert "dedupe key" in dup.message


def test_doctor_no_false_positive_when_content_differs(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/A")
    c1 = Chunk(node_path="WORK/A", content="fact one")
    c2 = Chunk(node_path="WORK/A", content="fact two")
    c1.content_hash = ""
    c2.content_hash = ""
    store.save_chunk(c1)
    store.save_chunk(c2)
    issues = Doctor(store).run()
    assert not any(i.check == "duplicate_active_chunk" for i in issues)
