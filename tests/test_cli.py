import json
import sys

from vertical_brain.cli.main import main
from vertical_brain.storage.json_store import JsonStore

STRUCTURED_SCHEMA_PATH = "WORK/DataArt/Databricks/Certification/StructuredStreaming/SchemaEvolution"
AUTO_LOADER_SCHEMA_PATH = "WORK/DataArt/Databricks/Certification/AutoLoader/SchemaEvolution"


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

    assert f"Target path: {STRUCTURED_SCHEMA_PATH}" in output
    store = JsonStore(tmp_path / "data")
    chunks = store.get_chunks_by_path(STRUCTURED_SCHEMA_PATH)
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

    assert f"Target path: {STRUCTURED_SCHEMA_PATH}" in output
    assert "Query type: explanation" in output
    assert "Allowed context policy: ancestors=True, peer_links=True, exclude_other_branches=True" in output
    assert "Answer:" in output
    assert "Based only on locked context:" in output
    assert "Databricks Delta schema evolution" in output


def test_cli_ingest_route_json_is_inspectable(monkeypatch, capsys, tmp_path):
    output = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "ingest",
        "--route-json",
        "Databricks Auto Loader schema evolution",
    )

    route_json = output.split("Route decision JSON:\n", maxsplit=1)[1].splitlines()[0]
    payload = json.loads(route_json)
    assert payload["target_path"] == AUTO_LOADER_SCHEMA_PATH
    assert payload["peer_links"][0]["path"] == STRUCTURED_SCHEMA_PATH


def test_cli_unknown_ingest_asks_clarification_without_writing_chunk(monkeypatch, capsys, tmp_path):
    output = run_cli(monkeypatch, capsys, tmp_path, "ingest", "random note")

    assert "Action: ask_clarification" in output
    assert "Clarification needed:" in output
    store = JsonStore(tmp_path / "data")
    assert store.list_chunks() == []


def test_cli_unknown_ask_requests_clarification(monkeypatch, capsys, tmp_path):
    output = run_cli(monkeypatch, capsys, tmp_path, "ask", "--route-json", "random question?")

    route_json = output.split("Route decision JSON:\n", maxsplit=1)[1].splitlines()[0]
    payload = json.loads(route_json)
    assert payload["target_path"] == "INBOX/Unclassified"
    assert payload["confidence"] == 0.4
    assert "Clarification needed:" in output
    assert "Allowed context:" not in output


def test_cli_uses_configured_data_dir(monkeypatch, capsys, tmp_path):
    output = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--data-dir",
        "brain-data",
        "ingest",
        "dbt ephemeral staging model",
    )

    assert "Target path: WORK/Stack/dbt" in output
    store = JsonStore(tmp_path / "brain-data")
    assert len(store.get_chunks_by_path("WORK/Stack/dbt")) == 1
    assert not (tmp_path / "data" / "chunks.json").exists()
