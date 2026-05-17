"""Version / dirty-flag propagation tests.

Verifies that every mutation that conceptually changes a branch also
increments the version of the affected node(s) and marks them dirty.
"""
from __future__ import annotations


from vertical_brain.core.models import Chunk, ChunkInput, LinkInput, StorageOperation
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.storage.json_store import JsonStore


# ── helper ────────────────────────────────────────────────────────────────────

def _executor(tmp_path):
    store = JsonStore(tmp_path)
    return store, StorageOperationExecutor(store)


# ── append_chunk ──────────────────────────────────────────────────────────────

def test_append_chunk_increments_target_version(tmp_path):
    store, ex = _executor(tmp_path)
    store.ensure_node("WORK/A")
    v0 = store.get_node("WORK/A").version
    ex.apply(StorageOperation(operation="append_chunk", target_path="WORK/A",
                               chunk=ChunkInput(content="fact")))
    assert store.get_node("WORK/A").version > v0


def test_append_chunk_marks_ancestor_versions(tmp_path):
    store, ex = _executor(tmp_path)
    store.ensure_node("WORK/A/Deep")
    v_work = store.get_node("WORK").version
    v_a = store.get_node("WORK/A").version
    ex.apply(StorageOperation(operation="append_chunk", target_path="WORK/A/Deep",
                               chunk=ChunkInput(content="fact")))
    assert store.get_node("WORK").version > v_work
    assert store.get_node("WORK/A").version > v_a


# ── mark_stale ────────────────────────────────────────────────────────────────

def test_mark_stale_increments_target_and_ancestor_versions(tmp_path):
    store, ex = _executor(tmp_path)
    result = ex.apply(StorageOperation(operation="append_chunk", target_path="WORK/A",
                                        chunk=ChunkInput(content="old fact")))
    v_work = store.get_node("WORK").version
    v_a = store.get_node("WORK/A").version

    ex.apply(StorageOperation(operation="mark_stale", target_path="WORK/A",
                               chunk_ids=[result.chunk_id]))
    assert store.get_node("WORK/A").version > v_a
    assert store.get_node("WORK").version > v_work


# ── supersede_chunk ───────────────────────────────────────────────────────────

def test_supersede_chunk_increments_target_version(tmp_path):
    store, ex = _executor(tmp_path)
    result = ex.apply(StorageOperation(operation="append_chunk", target_path="WORK/A",
                                        chunk=ChunkInput(content="fact")))
    v0 = store.get_node("WORK/A").version
    ex.apply(StorageOperation(operation="supersede_chunk", target_path="WORK/A",
                               chunk_ids=[result.chunk_id]))
    assert store.get_node("WORK/A").version > v0


# ── create_link ───────────────────────────────────────────────────────────────

def test_create_link_increments_source_branch_version(tmp_path):
    store, ex = _executor(tmp_path)
    store.ensure_node("WORK/A")
    store.ensure_node("WORK/B")
    v_a = store.get_node("WORK/A").version

    ex.apply(StorageOperation(
        operation="create_link",
        target_path="WORK/A",
        links=[LinkInput(target_path="WORK/B", link_type="peer", reason="related")],
    ))
    assert store.get_node("WORK/A").version > v_a


def test_create_link_increments_target_branch_version(tmp_path):
    store, ex = _executor(tmp_path)
    store.ensure_node("WORK/A")
    store.ensure_node("WORK/B")
    v_b = store.get_node("WORK/B").version

    ex.apply(StorageOperation(
        operation="create_link",
        target_path="WORK/A",
        links=[LinkInput(target_path="WORK/B", link_type="peer", reason="related")],
    ))
    assert store.get_node("WORK/B").version > v_b


# ── append_gold_aspect ────────────────────────────────────────────────────────

def test_append_gold_aspect_increments_target_version(tmp_path):
    store, ex = _executor(tmp_path)
    store.ensure_node("WORK/A")
    store.save_chunk(Chunk(node_path="WORK/A", content="silver summary", layer="silver"))
    v0 = store.get_node("WORK/A").version
    ex.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/A",
                               gold_aspect="key fact"))
    assert store.get_node("WORK/A").version > v0


def test_append_gold_aspect_increments_ancestor_version(tmp_path):
    store, ex = _executor(tmp_path)
    store.ensure_node("WORK/A")
    store.save_chunk(Chunk(node_path="WORK/A", content="silver summary", layer="silver"))
    v_work = store.get_node("WORK").version
    ex.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/A",
                               gold_aspect="key fact"))
    assert store.get_node("WORK").version > v_work


def test_append_gold_aspect_overflow_increments_overflow_path_version(tmp_path):
    from vertical_brain.core.gold import MAX_GOLD_ASPECTS, GoldAspect, serialize_gold_aspects
    store, ex = _executor(tmp_path)
    full_content = serialize_gold_aspects([GoldAspect(text=f"a{i}") for i in range(MAX_GOLD_ASPECTS)])
    from vertical_brain.core.models import Chunk
    store.save_chunk(Chunk(node_path="WORK/A", content="silver summary", layer="silver"))
    store.save_chunk(Chunk(node_path="WORK/A", content=full_content, layer="gold"))

    ex.apply(StorageOperation(operation="append_gold_aspect", target_path="WORK/A",
                               gold_aspect="overflow fact"))

    overflow_node = store.get_node("WORK/A_2")
    assert overflow_node is not None
    assert overflow_node.version > 0


# ── rename_namespace ──────────────────────────────────────────────────────────

def test_rename_namespace_marks_new_prefix_dirty(tmp_path):
    store, ex = _executor(tmp_path)
    store.ensure_node("OLD/Project")

    ex.apply(StorageOperation(operation="rename_namespace",
                               target_path="OLD/Project", new_path="NEW/Project"))

    new_node = store.get_node("NEW/Project")
    assert new_node is not None
    assert new_node.is_dirty


def test_rename_namespace_increments_new_prefix_ancestors(tmp_path):
    store, ex = _executor(tmp_path)
    store.ensure_node("OLD/Project")
    # Pre-create the new parent so _mark_ancestors_dirty can find and version it.
    store.ensure_node("NEW")
    v_new = store.get_node("NEW").version

    ex.apply(StorageOperation(operation="rename_namespace",
                               target_path="OLD/Project", new_path="NEW/Project"))

    assert store.get_node("NEW").version > v_new
