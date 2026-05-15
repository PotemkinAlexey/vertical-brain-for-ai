import json

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


def test_store_seeds_configured_root_namespaces(tmp_path):
    namespace_dir = tmp_path / "namespaces"
    namespace_dir.mkdir()
    (namespace_dir / "root.json").write_text(
        json.dumps({"roots": ["WORK", "PERSONAL", "TRADING", "INBOX"]}),
        encoding="utf-8",
    )

    store = JsonStore(tmp_path)

    assert [node.path for node in store.list_nodes()] == ["WORK", "PERSONAL", "TRADING", "INBOX"]


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


def test_save_link_does_not_duplicate_same_relationship(tmp_path):
    store = JsonStore(tmp_path)
    link = Link(
        source_path="WORK/DataArt/Databricks",
        target_path="WORK/DataArt/Databricks/AutoLoader",
        link_type="peer",
        reason="Related Databricks ingestion concepts.",
    )

    first = store.save_link(link)
    second = store.save_link(
        Link(
            source_path=link.source_path,
            target_path=link.target_path,
            link_type=link.link_type,
            reason="Repeated route decision.",
        )
    )

    assert first.id == second.id
    assert len(store.list_links()) == 1
    assert store.get_link(first.id) == first


def test_gold_aspect_append_writes_json_field_and_markdown_file(tmp_path):
    store = JsonStore(tmp_path)

    node = store.append_node_gold_aspect("WORK/DataArt", "Stable DataArt summary")

    assert node.gold_aspects == ["Stable DataArt summary"]
    gold_file = store.gold_summary_path("WORK/DataArt")
    assert gold_file.exists()
    assert gold_file.read_text(encoding="utf-8") == "# WORK/DataArt\n\n- Stable DataArt summary\n"
