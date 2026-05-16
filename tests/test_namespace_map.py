import pytest

from vertical_brain.core.context_session import ContextSession
from vertical_brain.core.models import Chunk, Link
from vertical_brain.storage.json_store import JsonStore
from vertical_brain.storage.sqlite_store import SQLiteStore


@pytest.mark.parametrize("store_class", [JsonStore, SQLiteStore])
def test_namespace_map_exposes_structure_counts_and_summaries_without_chunk_content(tmp_path, store_class):
    store = store_class(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="Stable DataArt summary", layer="gold"))
    store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/Databricks",
            content="Sensitive raw chunk content should not be in the map.",
        )
    )
    store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/Databricks",
            content="Old raw chunk content should not be in the map.",
            status="stale",
        )
    )
    store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/FXDB",
            content="Sibling raw chunk content should not be in the map.",
        )
    )
    store.save_link(
        Link(
            source_path="WORK/DataArt/Databricks",
            target_path="WORK/DataArt/FXDB",
            link_type="peer",
            reason="Related work branch.",
        )
    )

    namespace_map = ContextSession(store).namespace_map(root_path="WORK/DataArt")

    nodes = {node.path: node for node in namespace_map.nodes}
    assert namespace_map.root_path == "WORK/DataArt"
    assert set(nodes) == {"WORK/DataArt", "WORK/DataArt/Databricks", "WORK/DataArt/FXDB"}
    assert nodes["WORK/DataArt"].children == ["WORK/DataArt/Databricks", "WORK/DataArt/FXDB"]
    assert nodes["WORK/DataArt"].gold_summary == "Stable DataArt summary"
    assert nodes["WORK/DataArt"].subtree_chunk_count == 4
    assert nodes["WORK/DataArt/Databricks"].chunk_count == 2
    assert nodes["WORK/DataArt/Databricks"].active_chunk_count == 1
    assert nodes["WORK/DataArt/Databricks"].stale_chunk_count == 1
    assert nodes["WORK/DataArt/Databricks"].link_count == 1
    assert nodes["WORK/DataArt/Databricks"].link_handles[0].target_path == "WORK/DataArt/FXDB"
    assert "Sensitive raw chunk content" not in namespace_map.to_json()


def test_namespace_map_respects_relative_depth_limit_and_summary_budget(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="A" * 20, layer="gold"))
    store.ensure_node("WORK/DataArt/Databricks/Certification/SchemaEvolution")

    namespace_map = ContextSession(store).namespace_map(
        root_path="WORK/DataArt",
        max_depth=1,
        summary_max_chars=8,
    )

    nodes = {node.path: node for node in namespace_map.nodes}
    assert set(nodes) == {"WORK/DataArt", "WORK/DataArt/Databricks"}
    assert nodes["WORK/DataArt"].depth == 0
    assert nodes["WORK/DataArt/Databricks"].depth == 1
    assert nodes["WORK/DataArt"].gold_summary == "AAAAA..."
    assert nodes["WORK/DataArt"].omitted_summary_chars == 12
    assert namespace_map.omitted_nodes == 2


def test_namespace_map_indexes_chunks_and_links_once_per_build(tmp_path):
    class CountingList(list):
        def __init__(self, rows):
            super().__init__(rows)
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return super().__iter__()

    store = JsonStore(tmp_path)
    for index in range(25):
        store.save_chunk(Chunk(node_path=f"WORK/Project/Node{index}", content=f"fact {index}"))
    store.save_link(Link(
        source_path="WORK/Project/Node0",
        target_path="WORK/Project/Node1",
        link_type="peer",
        reason="related",
    ))

    chunks = CountingList(store.list_chunks())
    links = CountingList(store.list_links())
    store.list_chunks = lambda: chunks  # type: ignore[method-assign]
    store.list_links = lambda: links  # type: ignore[method-assign]

    namespace_map = ContextSession(store).namespace_map(root_path="WORK/Project")

    nodes = {node.path: node for node in namespace_map.nodes}
    assert len(nodes) == 26
    assert nodes["WORK/Project"].subtree_chunk_count == 25
    assert chunks.iterations <= 3
    assert links.iterations == 1
