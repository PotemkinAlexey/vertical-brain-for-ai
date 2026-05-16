"""Optimizer snapshot read isolation tests.

Acceptance criteria:
- decay disabled: get_chunks_by_path called once, list_links never called.
- decay enabled:  get_chunks_by_path called once, list_links called once.
- _build_plan:    does not touch the store at all.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch


from vertical_brain.core.models import Chunk, Link
from vertical_brain.core.optimizer import SimpleOptimizer
from vertical_brain.storage.json_store import JsonStore


MIN_PARTS = 3


def _old_chunk(path: str, content: str = "old fact") -> Chunk:
    c = Chunk(node_path=path, content=content)
    c.created_at = (datetime.now(timezone.utc) - timedelta(days=91)).isoformat()
    return c


# ── Item 1: explicit snapshot reads ──────────────────────────────────────────

def test_optimize_branch_decay_disabled_reads_chunks_once_no_links(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(_old_chunk("WORK/A/B", "fact"))

    with patch.object(store, "get_chunks_by_path", wraps=store.get_chunks_by_path) as mock_chunks:
        with patch.object(store, "list_links", wraps=store.list_links) as mock_links:
            opt = SimpleOptimizer(store, min_compaction_path_parts=MIN_PARTS, decay_rate=1.0)
            opt.optimize_branch("WORK/A")

    mock_chunks.assert_called_once()
    mock_links.assert_not_called()


def test_optimize_branch_decay_enabled_reads_chunks_once_links_once(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(_old_chunk("WORK/A/B", "fact"))

    with patch.object(store, "get_chunks_by_path", wraps=store.get_chunks_by_path) as mock_chunks:
        with patch.object(store, "list_links", wraps=store.list_links) as mock_links:
            opt = SimpleOptimizer(
                store, min_compaction_path_parts=MIN_PARTS, decay_rate=0.5, decay_days=30
            )
            opt.optimize_branch("WORK/A")

    mock_chunks.assert_called_once()
    mock_links.assert_called_once()


def test_plan_branch_decay_disabled_reads_chunks_once_no_links(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/A/B")

    with patch.object(store, "get_chunks_by_path", wraps=store.get_chunks_by_path) as mock_chunks:
        with patch.object(store, "list_links", wraps=store.list_links) as mock_links:
            opt = SimpleOptimizer(store, min_compaction_path_parts=MIN_PARTS, decay_rate=1.0)
            opt.plan_branch("WORK/A/B")

    mock_chunks.assert_called_once()
    mock_links.assert_not_called()


def test_plan_branch_decay_enabled_reads_chunks_once_links_once(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/A/B")

    with patch.object(store, "get_chunks_by_path", wraps=store.get_chunks_by_path) as mock_chunks:
        with patch.object(store, "list_links", wraps=store.list_links) as mock_links:
            opt = SimpleOptimizer(
                store, min_compaction_path_parts=MIN_PARTS, decay_rate=0.5, decay_days=30
            )
            opt.plan_branch("WORK/A/B")

    mock_chunks.assert_called_once()
    mock_links.assert_called_once()


def test_build_plan_does_not_access_store():
    """_build_plan must be pure — no store attribute access."""
    store = MagicMock()
    opt = SimpleOptimizer(store, min_compaction_path_parts=MIN_PARTS, decay_rate=0.5, decay_days=30)
    chunk = _old_chunk("WORK/A/B")
    opt._build_plan([chunk], "WORK/A", linked_paths=set())
    store.assert_not_called()
    # Verify none of the store's attributes were accessed inside _build_plan.
    for attr_name in ("list_chunks", "list_links", "get_chunks_by_path", "get_node"):
        assert not getattr(store, attr_name).called, f"store.{attr_name} was called inside _build_plan"


# ── Item 2: subtree-aware decay protection ────────────────────────────────────

def test_decay_protects_child_of_linked_path(tmp_path):
    """Link on WORK/A must protect WORK/A/B chunk from decay."""
    store = JsonStore(tmp_path)
    chunk = _old_chunk("WORK/A/B")
    store.save_chunk(chunk)
    store.save_link(Link(
        source_path="WORK/A", target_path="WORK/Other", link_type="peer", reason="linked"
    ))

    opt = SimpleOptimizer(
        store, min_compaction_path_parts=MIN_PARTS, decay_rate=0.5, decay_days=30, stale_threshold=0.2
    )
    opt.optimize_branch("WORK/A")

    result = store.get_chunks_by_path("WORK/A/B")
    assert all(c.status == "active" for c in result)


def test_decay_does_not_protect_parent_of_linked_path(tmp_path):
    """Link on WORK/A/B must NOT protect the WORK/A parent chunk."""
    store = JsonStore(tmp_path)
    parent_chunk = _old_chunk("WORK/A")
    parent_chunk.confidence = 0.99
    store.save_chunk(parent_chunk)
    store.save_link(Link(
        source_path="WORK/A/B", target_path="WORK/Other", link_type="peer", reason="linked"
    ))

    opt = SimpleOptimizer(
        store, min_compaction_path_parts=2, decay_rate=0.5, decay_days=30, stale_threshold=0.2
    )
    opt.optimize_branch("WORK")

    result = store.get_chunks_by_path("WORK/A")
    assert any(c.status == "stale" for c in result)


def test_decay_unrelated_link_does_not_protect_chunk(tmp_path):
    """A link between WORK/X and WORK/Y must not protect WORK/A/B."""
    store = JsonStore(tmp_path)
    chunk = _old_chunk("WORK/A/B")
    store.save_chunk(chunk)
    store.save_link(Link(
        source_path="WORK/X", target_path="WORK/Y", link_type="peer", reason="unrelated"
    ))

    opt = SimpleOptimizer(
        store, min_compaction_path_parts=MIN_PARTS, decay_rate=0.5, decay_days=30, stale_threshold=0.2
    )
    opt.optimize_branch("WORK/A")

    result = store.get_chunks_by_path("WORK/A/B")
    assert any(c.status == "stale" for c in result)
