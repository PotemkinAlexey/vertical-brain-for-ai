import pytest

from vertical_brain.core.context_lock import ContextLock
from vertical_brain.core.models import Link
from vertical_brain.storage.json_store import JsonStore


def test_peer_link_expansion_works_without_from_path(tmp_path):
    store = JsonStore(tmp_path)
    link = store.save_link(Link(
        source_path="WORK/A",
        target_path="WORK/B",
        link_type="peer",
        reason="related work",
    ))
    resolved = ContextLock(store).expand_link(link.id)
    assert resolved == "WORK/B"


def test_peer_link_expands_in_reverse_when_from_path_is_target(tmp_path):
    store = JsonStore(tmp_path)
    link = store.save_link(Link(
        source_path="WORK/A",
        target_path="WORK/B",
        link_type="peer",
        reason="related work",
    ))
    resolved = ContextLock(store).expand_link(link.id, from_path="WORK/B")
    assert resolved == "WORK/A"


def test_directional_link_expansion_without_from_path_raises(tmp_path):
    store = JsonStore(tmp_path)
    link = store.save_link(Link(
        source_path="WORK/A",
        target_path="WORK/B",
        link_type="derived_from",
        reason="A derived from B",
    ))
    with pytest.raises(ValueError, match="from_path is required"):
        ContextLock(store).expand_link(link.id)


def test_directional_link_expansion_with_from_path_works(tmp_path):
    store = JsonStore(tmp_path)
    link = store.save_link(Link(
        source_path="WORK/A",
        target_path="WORK/B",
        link_type="derived_from",
        reason="A derived from B",
    ))
    lock = ContextLock(store)
    assert lock.expand_link(link.id, from_path="WORK/A") == "WORK/B"
    assert lock.expand_link(link.id, from_path="WORK/B") == "WORK/A"


def test_directional_link_rejects_unrelated_from_path(tmp_path):
    store = JsonStore(tmp_path)
    link = store.save_link(Link(
        source_path="WORK/A",
        target_path="WORK/B",
        link_type="gold_overflow",
        reason="overflow chain",
    ))
    with pytest.raises(ValueError, match="not an endpoint"):
        ContextLock(store).expand_link(link.id, from_path="WORK/UNRELATED")
