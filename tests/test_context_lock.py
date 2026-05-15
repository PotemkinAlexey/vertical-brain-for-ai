from vertical_brain.core.context_lock import ContextLock
from vertical_brain.core.models import Chunk, Link
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


def test_context_lock_includes_ancestor_summaries_and_peer_links(tmp_path):
    store = JsonStore(tmp_path)
    store.update_node_gold_summary("WORK/DataArt", "DataArt gold summary")
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
    context = lock.build_context("WORK/DataArt/Databricks")

    joined = "\n".join(context)
    assert "DataArt gold summary" in joined
    assert "Databricks target fact" in joined
    assert "Auto Loader peer fact" in joined
    assert "FXDB unrelated fact" not in joined


def test_context_lock_respects_context_policy_flags(tmp_path):
    store = JsonStore(tmp_path)
    store.update_node_gold_summary("WORK/DataArt", "DataArt gold summary")
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
