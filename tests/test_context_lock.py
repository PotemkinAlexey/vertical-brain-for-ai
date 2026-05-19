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
    chunk_a = Chunk(node_path="WORK/DataArt/Databricks", content="First fact")
    chunk_b = Chunk(node_path="WORK/DataArt/Databricks", content="Second fact")
    store.save_chunk(chunk_a)
    store.save_chunk(chunk_b)

    locked_context = ContextLock(store).open_locked_context(
        "WORK/DataArt/Databricks",
        budget=ContextBudget(max_items=1),
    )

    assert len(locked_context.items) == 1
    assert locked_context.omitted_items == 1
    # v1.8: omitted chunk_ids are surfaced so the agent can fetch dropped
    # evidence with `list_chunks` instead of re-running `read_context`.
    assert locked_context.omitted_chunk_ids == [chunk_b.id]


# ── cycle guard ───────────────────────────────────────────────────────────────

def test_context_lock_expanded_links_do_not_loop_on_mutual_peer(tmp_path):
    """A↔B peer links with link_expansion='expanded' must not recurse infinitely."""
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="Fact about A"))
    store.save_chunk(Chunk(node_path="WORK/B", content="Fact about B"))
    store.save_link(Link(source_path="WORK/A", target_path="WORK/B", link_type="peer", reason="related"))
    store.save_link(Link(source_path="WORK/B", target_path="WORK/A", link_type="peer", reason="related"))

    locked = ContextLock(store).open_locked_context(
        "WORK/A",
        policy=ContextPolicy(link_expansion="expanded"),
    )

    contents = [item.content for item in locked.items]
    assert "Fact about A" in contents
    assert "Fact about B" in contents
    assert contents.count("Fact about B") == 1  # not duplicated due to cycle


def test_context_lock_expanded_links_skip_target_path_itself(tmp_path):
    """If a link handle points back to the target path, it must not be re-expanded."""
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="Fact about A"))
    store.save_link(Link(source_path="WORK/A", target_path="WORK/A", link_type="peer", reason="self ref"))

    locked = ContextLock(store).open_locked_context(
        "WORK/A",
        policy=ContextPolicy(link_expansion="expanded"),
    )

    contents = [item.content for item in locked.items]
    assert contents.count("Fact about A") == 1


# ── context priority stratification ──────────────────────────────────────────

def test_context_lock_gold_chunks_fill_budget_before_bronze(tmp_path):
    """When budget is tight, Gold items must be included before Bronze."""
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="bronze fact one", layer="bronze"))
    store.save_chunk(Chunk(node_path="WORK/A", content="bronze fact two", layer="bronze"))
    store.save_chunk(Chunk(node_path="WORK/A", content="gold summary", layer="gold"))

    locked = ContextLock(store).open_locked_context(
        "WORK/A",
        policy=ContextPolicy(include_ancestors=False),
        budget=ContextBudget(max_items=2),
    )

    layers = [item.layer for item in locked.items]
    assert "gold" in layers
    assert locked.omitted_items >= 1


def test_context_lock_silver_fills_before_bronze(tmp_path):
    """Silver chunks must be included before Bronze when budget is limited."""
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="bronze raw note", layer="bronze"))
    store.save_chunk(Chunk(node_path="WORK/A", content="silver compact fact", layer="silver"))
    store.save_chunk(Chunk(node_path="WORK/A", content="another bronze note", layer="bronze"))

    locked = ContextLock(store).open_locked_context(
        "WORK/A",
        policy=ContextPolicy(include_ancestors=False),
        budget=ContextBudget(max_items=1),
    )

    assert locked.items[0].layer == "silver"
    assert locked.omitted_items == 2


def test_context_lock_does_not_duplicate_target_gold(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="gold summary", layer="gold"))
    store.save_chunk(Chunk(node_path="WORK/A", content="bronze fact", layer="bronze"))

    locked = ContextLock(store).open_locked_context(
        "WORK/A",
        policy=ContextPolicy(include_ancestors=False),
    )

    contents = [(item.layer, item.content) for item in locked.items]
    assert contents.count(("gold", "gold summary")) == 1
    assert ("bronze", "bronze fact") in contents
