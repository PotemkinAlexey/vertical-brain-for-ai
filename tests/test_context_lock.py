import pytest

from vertical_brain.core.context_lock import ContextLock
from vertical_brain.core.models import Chunk, ContextBudget, ContextPolicy, Link
from vertical_brain.storage.json_store import JsonStore


def test_context_lock_excludes_other_branches(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Databricks fact"))
    store.save_chunk(Chunk(node_path="TRADING/Bots", content="Trading bot fact"))

    lock = ContextLock(store)
    context = lock.build_context("WORK/DataArt/Databricks")

    joined = "\n".join(context)
    assert "Databricks fact" in joined
    assert "Trading bot fact" not in joined


def test_context_lock_returns_locked_context_with_link_handles_only_by_default(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="DataArt gold summary", layer="gold"))
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Databricks target fact"))
    store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/Databricks/AutoLoader",
            content="Auto Loader peer fact",
        )
    )
    store.save_chunk(Chunk(node_path="WORK/DataArt/FXDB", content="FXDB unrelated fact"))
    store.save_link(
        Link(
            source_path="WORK/DataArt/Databricks",
            target_path="WORK/DataArt/Databricks/AutoLoader",
            link_type="peer",
            reason="Schema evolution concepts are related.",
        )
    )

    lock = ContextLock(store)
    locked_context = lock.open_locked_context("WORK/DataArt/Databricks")

    joined = "\n".join(locked_context.as_prompt_lines())
    assert "DataArt gold summary" in joined
    assert "Databricks target fact" in joined
    assert "Auto Loader peer fact" not in joined
    assert "FXDB unrelated fact" not in joined
    assert len(locked_context.link_handles) == 1
    assert locked_context.link_handles[0].target_path == "WORK/DataArt/Databricks/AutoLoader"


def test_context_lock_respects_context_policy_flags(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="DataArt gold summary", layer="gold"))
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Databricks target fact"))
    store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/Databricks/AutoLoader",
            content="Auto Loader peer fact",
        )
    )
    store.save_link(
        Link(
            source_path="WORK/DataArt/Databricks",
            target_path="WORK/DataArt/Databricks/AutoLoader",
            link_type="peer",
            reason="Schema evolution concepts are related.",
        )
    )

    lock = ContextLock(store)
    context = lock.build_context(
        "WORK/DataArt/Databricks",
        include_ancestors=False,
        include_peer_links=False,
    )

    joined = "\n".join(context)
    assert "Databricks target fact" in joined
    assert "DataArt gold summary" not in joined
    assert "Auto Loader peer fact" not in joined


def test_context_lock_can_expand_links_explicitly(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Databricks target fact"))
    store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/Databricks/AutoLoader",
            content="Auto Loader peer fact",
        )
    )
    store.save_link(
        Link(
            source_path="WORK/DataArt/Databricks",
            target_path="WORK/DataArt/Databricks/AutoLoader",
            link_type="peer",
            reason="Schema evolution concepts are related.",
        )
    )

    locked_context = ContextLock(store).open_locked_context(
        "WORK/DataArt/Databricks",
        policy=ContextPolicy(link_expansion="expanded"),
    )

    joined = "\n".join(locked_context.as_prompt_lines())
    assert "Databricks target fact" in joined
    assert "Auto Loader peer fact" in joined


def test_context_lock_respects_budget(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="First fact"))
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="Second fact"))

    locked_context = ContextLock(store).open_locked_context(
        "WORK/DataArt/Databricks",
        budget=ContextBudget(max_items=1),
    )

    assert len(locked_context.items) == 1
    assert locked_context.omitted_items == 1


# ── ContextLock.expand_link ───────────────────────────────────────────────────

def test_expand_link_peer_works_without_from_path(tmp_path):
    store = JsonStore(tmp_path)
    link = store.save_link(Link(
        source_path="WORK/A", target_path="WORK/B",
        link_type="peer", reason="related",
    ))
    assert ContextLock(store).expand_link(link.id) == "WORK/B"


def test_expand_link_peer_reverse_with_from_path_target(tmp_path):
    store = JsonStore(tmp_path)
    link = store.save_link(Link(
        source_path="WORK/A", target_path="WORK/B",
        link_type="peer", reason="related",
    ))
    assert ContextLock(store).expand_link(link.id, from_path="WORK/B") == "WORK/A"


def test_expand_link_gold_overflow_raises_without_from_path(tmp_path):
    store = JsonStore(tmp_path)
    link = store.save_link(Link(
        source_path="WORK/Databricks", target_path="WORK/Databricks_2",
        link_type="gold_overflow", reason="Gold capacity exceeded",
    ))
    with pytest.raises(ValueError, match="from_path is required"):
        ContextLock(store).expand_link(link.id)


def test_expand_link_gold_overflow_source_to_target(tmp_path):
    store = JsonStore(tmp_path)
    link = store.save_link(Link(
        source_path="WORK/Databricks", target_path="WORK/Databricks_2",
        link_type="gold_overflow", reason="Gold capacity exceeded",
    ))
    assert ContextLock(store).expand_link(link.id, from_path="WORK/Databricks") == "WORK/Databricks_2"


def test_expand_link_gold_overflow_target_to_source(tmp_path):
    store = JsonStore(tmp_path)
    link = store.save_link(Link(
        source_path="WORK/Databricks", target_path="WORK/Databricks_2",
        link_type="gold_overflow", reason="Gold capacity exceeded",
    ))
    assert ContextLock(store).expand_link(link.id, from_path="WORK/Databricks_2") == "WORK/Databricks"


def test_expand_link_gold_overflow_rejects_unrelated_from_path(tmp_path):
    store = JsonStore(tmp_path)
    link = store.save_link(Link(
        source_path="WORK/Databricks", target_path="WORK/Databricks_2",
        link_type="gold_overflow", reason="Gold capacity exceeded",
    ))
    with pytest.raises(ValueError, match="not an endpoint"):
        ContextLock(store).expand_link(link.id, from_path="WORK/Python")
