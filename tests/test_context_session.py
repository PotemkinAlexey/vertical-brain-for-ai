from vertical_brain.core.context_session import ContextSession
from vertical_brain.core.models import Chunk, Link
from vertical_brain.storage.json_store import JsonStore


def test_context_session_uses_search_as_navigation_to_locked_context(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="DataArt ancestor summary", layer="gold"))
    store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/Databricks/StructuredStreaming",
            content="mergeSchema mergeSchema mergeSchema target fact.",
        )
    )
    store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/Databricks/StructuredStreaming",
            content="Additional target context without the query term.",
        )
    )
    store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/Databricks/AutoLoader",
            content="mergeSchema omitted candidate fact.",
        )
    )

    result = ContextSession(store).search_locked_context(
        "mergeSchema",
        root_path="WORK/DataArt",
        search_limit=10,
        context_limit=1,
        items_per_context=4,
    )

    assert len(result.candidate_handles) == 2
    assert "snippet" not in result.candidate_handles[0].to_dict()
    assert len(result.locked_contexts) == 1
    assert result.locked_contexts[0].target_path == "WORK/DataArt/Databricks/StructuredStreaming"
    locked_lines = "\n".join(result.locked_contexts[0].as_prompt_lines())
    assert "DataArt ancestor summary" in locked_lines
    assert "mergeSchema mergeSchema mergeSchema target fact." in locked_lines
    assert "Additional target context without the query term." in locked_lines
    assert "omitted candidate fact" not in locked_lines
    assert result.omitted_candidates == 1


def test_context_session_exposes_horizontal_links_as_handles_by_default(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/Databricks/StructuredStreaming",
            content="schema evolution target fact.",
        )
    )
    store.save_chunk(
        Chunk(
            node_path="WORK/DataArt/Databricks/AutoLoader",
            content="linked Auto Loader content should stay locked.",
        )
    )
    store.save_link(
        Link(
            source_path="WORK/DataArt/Databricks/StructuredStreaming",
            target_path="WORK/DataArt/Databricks/AutoLoader",
            link_type="peer",
            reason="Related schema evolution concept.",
        )
    )

    result = ContextSession(store).search_locked_context("schema evolution")

    locked_context = result.locked_contexts[0]
    locked_lines = "\n".join(locked_context.as_prompt_lines())
    assert "schema evolution target fact." in locked_lines
    assert "linked Auto Loader content should stay locked." not in locked_lines
    assert len(locked_context.link_handles) == 1
    assert locked_context.link_handles[0].target_path == "WORK/DataArt/Databricks/AutoLoader"
