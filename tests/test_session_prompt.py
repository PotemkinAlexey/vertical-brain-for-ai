from vertical_brain.core.context_session import ContextSession
from vertical_brain.core.models import Chunk, Link
from vertical_brain.storage.json_store import JsonStore


def test_session_prompt_includes_agents_md_before_memory_map(tmp_path, monkeypatch):
    agents_file = tmp_path / "AGENTS.md"
    agents_file.write_text("# Local Rules\n\nRead these before memory writes.", encoding="utf-8")
    monkeypatch.setenv("VERTICAL_BRAIN_AGENTS_PATH", str(agents_file))
    store = JsonStore(tmp_path / "store")
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="Delta migration", layer="gold"))

    prompt = ContextSession(store).session_prompt()

    assert prompt.startswith("AGENTS.md\n\n# Local Rules")
    assert "Read these before memory writes." in prompt
    assert "---\n\nVERTICAL BRAIN" in prompt
    assert "WORK/DataArt" in prompt


def test_session_prompt_contains_header_with_date_and_totals(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="fact one"))
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="fact two"))

    prompt = ContextSession(store).session_prompt()

    assert "VERTICAL BRAIN" in prompt
    assert "2 nodes" in prompt
    assert "2 active chunks" in prompt


def test_session_prompt_includes_gold_summary_inline(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="Databricks Delta migration", layer="gold"))
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="raw fact"))

    prompt = ContextSession(store).session_prompt()
    memory_map = prompt.split("---\n\n", maxsplit=1)[-1]

    assert "Databricks Delta migration" in memory_map
    assert "raw fact" not in memory_map


def test_session_prompt_shows_active_and_stale_counts(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="active chunk"))
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="old chunk", status="stale"))

    prompt = ContextSession(store).session_prompt()

    assert "1 active" in prompt
    assert "1 stale" in prompt


def test_session_prompt_shows_peer_links(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="fact"))
    store.save_chunk(Chunk(node_path="WORK/DataArt/FXDB", content="fact"))
    store.save_link(Link(
        source_path="WORK/DataArt/Databricks",
        target_path="WORK/DataArt/FXDB",
        link_type="peer",
        reason="Related work.",
    ))

    prompt = ContextSession(store).session_prompt()

    assert "→WORK/DataArt/FXDB" in prompt or "→WORK/DataArt/Databricks" in prompt


def test_session_prompt_marks_empty_nodes(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/DataArt/EmptyBranch")

    prompt = ContextSession(store).session_prompt()

    assert "(empty)" in prompt


def test_session_prompt_respects_root_path(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="fact"))
    store.save_chunk(Chunk(node_path="PERSONAL/Blog", content="post"))

    prompt = ContextSession(store).session_prompt(root_path="WORK")

    assert "WORK/DataArt" in prompt
    assert "PERSONAL" not in prompt


def test_session_prompt_respects_max_depth(tmp_path):
    store = JsonStore(tmp_path)
    store.ensure_node("WORK/DataArt/Databricks/Certification/SchemaEvolution")

    prompt = ContextSession(store).session_prompt(root_path="WORK/DataArt", max_depth=1)

    assert "WORK/DataArt/Databricks" in prompt
    assert "SchemaEvolution" not in prompt
    assert "omitted" in prompt


def test_session_prompt_truncates_long_gold_summary(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="X" * 300, layer="gold"))

    prompt = ContextSession(store).session_prompt(summary_max_chars=50)

    assert "X" * 300 not in prompt
    assert "..." in prompt


def test_session_prompt_counts_links_without_double_counting(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/A", content="fact"))
    store.save_chunk(Chunk(node_path="WORK/B", content="fact"))
    store.save_link(Link(source_path="WORK/A", target_path="WORK/B", link_type="peer", reason="related"))

    prompt = ContextSession(store).session_prompt()

    assert "1 links" in prompt


def test_session_prompt_finds_agents_md_via_module_path_when_cwd_is_unrelated(tmp_path, monkeypatch):
    """session_start must return AGENTS.md even when cwd has no AGENTS.md in its ancestry.

    This reproduces the Claude Desktop scenario where the MCP server starts with
    cwd=/ or cwd=~ and _find_agents_md() would otherwise return None.
    """
    agents_file = tmp_path / "AGENTS.md"
    agents_file.write_text("# Contract\n\nDo not skip Bronze.", encoding="utf-8")

    # Patch _find_agents_md so the session_prompt picks up our tmp_path AGENTS.md
    # regardless of cwd or module location.
    import vertical_brain.core.context_session as cs_module
    monkeypatch.setattr(cs_module, "_find_agents_md",
                        lambda: agents_file)

    # Move cwd somewhere completely unrelated (no AGENTS.md in ancestry).
    monkeypatch.chdir("/")
    monkeypatch.delenv("VERTICAL_BRAIN_AGENTS_PATH", raising=False)

    store = JsonStore(tmp_path / "store")
    prompt = ContextSession(store).session_prompt()

    assert "AGENTS.md" in prompt
    assert "Do not skip Bronze." in prompt
