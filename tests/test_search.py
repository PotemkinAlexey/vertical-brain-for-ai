import pytest

from vertical_brain.core.models import Chunk
from vertical_brain.core.search import BrainSearch
from vertical_brain.storage.json_store import JsonStore
from vertical_brain.storage.sqlite_store import SQLiteStore


@pytest.mark.parametrize("store_class", [JsonStore, SQLiteStore])
def test_search_finds_active_chunks_inside_requested_branch(tmp_path, store_class):
    store = store_class(tmp_path)
    store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/Databricks/Certification",
            content="Delta streaming sink schema evolution uses mergeSchema.",
            layer="silver",
            content_type="fact",
        )
    )
    store.save_chunk(
        Chunk(
            node_path="WORK/Other/Databricks",
            content="mergeSchema exists outside the requested branch.",
        )
    )

    results = BrainSearch(store).search("mergeSchema", root_path="WORK/DataArt", limit=5)

    assert len(results) == 1
    assert results[0].path == "WORK/DataArt/Databricks/Certification"
    assert results[0].source == "chunk"
    assert "mergeSchema" in results[0].snippet


@pytest.mark.parametrize("store_class", [JsonStore, SQLiteStore])
def test_search_excludes_stale_chunks_by_default(tmp_path, store_class):
    store = store_class(tmp_path)
    store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/Databricks",
            content="Legacy schema evolution note.",
            status="stale",
        )
    )

    active_results = BrainSearch(store).search("schema evolution")
    all_results = BrainSearch(store).search("schema evolution", include_stale=True)

    assert active_results == []
    assert len(all_results) == 1
    assert all_results[0].status == "stale"


@pytest.mark.parametrize("store_class", [JsonStore, SQLiteStore])
def test_search_finds_gold_summaries(tmp_path, store_class):
    store = store_class(tmp_path)
    store.update_node_gold_summary(
        "WORK/DataArt/Databricks",
        "Canonical schema evolution summary for Databricks certification.",
    )

    results = BrainSearch(store).search("canonical")

    assert len(results) == 1
    assert results[0].source == "gold"
    assert results[0].path == "WORK/DataArt/Databricks"


def test_sqlite_search_index_tracks_chunk_updates(tmp_path):
    store = SQLiteStore(tmp_path)
    chunk = store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/Databricks",
            content="Original searchable note.",
        )
    )
    chunk.content = "Updated schema evolution note."
    chunk.status = "stale"
    store.update_chunk(chunk)

    active_results = BrainSearch(store).search("updated")
    stale_results = BrainSearch(store).search("updated", include_stale=True)

    assert active_results == []
    assert len(stale_results) == 1
    assert stale_results[0].chunk_id == chunk.id
