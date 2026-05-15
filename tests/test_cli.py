import sys

from vertical_brain.cli.main import main
from vertical_brain.storage.json_store import JsonStore


def run_cli(monkeypatch, capsys, tmp_path, *args):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vb", *args])
    main()
    return capsys.readouterr().out


def test_cli_ingest_writes_chunk_and_tree_lists_namespace(monkeypatch, capsys, tmp_path):
    output = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "ingest",
        "Databricks Delta schema evolution",
    )

    assert "Target path: WORK/DataArt/Databricks" in output
    store = JsonStore(tmp_path / "data")
    chunks = store.get_chunks_by_path("WORK/DataArt/Databricks")
    assert len(chunks) == 1
    assert chunks[0].content == "Databricks Delta schema evolution"

    tree_output = run_cli(monkeypatch, capsys, tmp_path, "tree")
    assert "WORK" in tree_output
    assert "DataArt" in tree_output
    assert "Databricks" in tree_output


def test_cli_ask_prints_query_route_contract(monkeypatch, capsys, tmp_path):
    run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "ingest",
        "Databricks Delta schema evolution",
    )

    output = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "ask",
        "How does Databricks schema evolution work?",
    )

    assert "Target path: WORK/DataArt/Databricks" in output
    assert "Query type: explanation" in output
    assert "Allowed context policy: ancestors=True, peer_links=True, exclude_other_branches=True" in output
    assert "Answer:" in output
    assert "Based only on locked context:" in output
    assert "Databricks Delta schema evolution" in output
