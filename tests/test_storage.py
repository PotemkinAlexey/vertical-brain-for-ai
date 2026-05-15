from vertical_brain.core.models import Link
from vertical_brain.storage.json_store import JsonStore


def test_ensure_node_creates_ancestor_chain(tmp_path):
    store = JsonStore(tmp_path)

    store.ensure_node("WORK/DataArt/Databricks")

    nodes = {node.path: node for node in store.list_nodes()}
    assert set(nodes) == {"WORK", "WORK/DataArt", "WORK/DataArt/Databricks"}
    assert nodes["WORK"].parent_path is None
    assert nodes["WORK/DataArt"].parent_path == "WORK"
    assert nodes["WORK/DataArt/Databricks"].parent_path == "WORK/DataArt"


def test_peer_paths_are_available_from_both_sides(tmp_path):
    store = JsonStore(tmp_path)
    store.save_link(
        Link(
            source_path="WORK/DataArt/Databricks",
            target_path="WORK/DataArt/Databricks/AutoLoader",
            link_type="peer",
            reason="Related Databricks ingestion concepts.",
        )
    )

    assert store.get_peer_paths("WORK/DataArt/Databricks") == [
        "WORK/DataArt/Databricks/AutoLoader"
    ]
    assert store.get_peer_paths("WORK/DataArt/Databricks/AutoLoader") == [
        "WORK/DataArt/Databricks"
    ]


def test_gold_summary_update_writes_json_field_and_markdown_file(tmp_path):
    store = JsonStore(tmp_path)

    node = store.update_node_gold_summary("WORK/DataArt", "Stable DataArt summary")

    assert node.gold_summary == "Stable DataArt summary"
    gold_file = store.gold_summary_path("WORK/DataArt")
    assert gold_file.exists()
    assert gold_file.read_text(encoding="utf-8") == "# WORK/DataArt\n\nStable DataArt summary\n"
