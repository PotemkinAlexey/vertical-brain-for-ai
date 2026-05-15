import json
import sys

import pytest

from vertical_brain.cli.main import main
from vertical_brain.storage.json_store import JsonStore
from vertical_brain.storage.sqlite_store import SQLiteStore

STRUCTURED_SCHEMA_PATH = "WORK/DataArt/Databricks/Certification/StructuredStreaming/SchemaEvolution"
AUTO_LOADER_SCHEMA_PATH = "WORK/DataArt/Databricks/Certification/AutoLoader/SchemaEvolution"


def ingest_response(**overrides):
    payload = {
        "target_path": STRUCTURED_SCHEMA_PATH,
        "content_type": "fact",
        "layer": "silver",
        "action": "append_and_optimize",
        "peer_links": [],
        "stale_candidates": [],
        "confidence": 0.9,
        "reasoning_summary": "Model selected the route.",
    }
    payload.update(overrides)
    return payload


def query_response(**overrides):
    payload = {
        "target_path": STRUCTURED_SCHEMA_PATH,
        "allowed_context": {
            "include_ancestors": True,
            "include_peer_links": True,
            "exclude_other_branches": True,
        },
        "query_type": "explanation",
        "confidence": 0.9,
        "reasoning_summary": "Model selected the query route.",
    }
    payload.update(overrides)
    return payload


def write_llm_response(tmp_path, name, payload):
    response_file = tmp_path / name
    response_file.write_text(json.dumps(payload), encoding="utf-8")
    return response_file


def run_cli(monkeypatch, capsys, tmp_path, *args):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vb", *args])
    main()
    return capsys.readouterr().out


def test_cli_ingest_writes_chunk_and_tree_lists_namespace(monkeypatch, capsys, tmp_path):
    response_file = write_llm_response(tmp_path, "ingest.json", ingest_response())
    output = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(response_file),
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
    ingest_response_file = write_llm_response(tmp_path, "ingest.json", ingest_response())
    run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(ingest_response_file),
        "ingest",
        "Databricks Delta schema evolution",
    )

    query_response_file = write_llm_response(tmp_path, "query.json", query_response())
    output = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(query_response_file),
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
    response_file = write_llm_response(
        tmp_path,
        "ingest_auto_loader.json",
        ingest_response(
            target_path=AUTO_LOADER_SCHEMA_PATH,
            peer_links=[
                {
                    "path": STRUCTURED_SCHEMA_PATH,
                    "reason": "Model approved this peer.",
                }
            ],
        ),
    )
    output = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(response_file),
        "ingest",
        "--route-json",
        "Databricks Auto Loader schema evolution",
    )

    route_json = output.split("Route decision JSON:\n", maxsplit=1)[1].splitlines()[0]
    payload = json.loads(route_json)
    assert payload["target_path"] == AUTO_LOADER_SCHEMA_PATH
    assert payload["peer_links"][0]["path"] == STRUCTURED_SCHEMA_PATH


def test_cli_ingest_prints_stale_candidates_without_mutating_old_chunks(monkeypatch, capsys, tmp_path):
    first_response_file = write_llm_response(tmp_path, "first_ingest.json", ingest_response())
    run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(first_response_file),
        "ingest",
        "Databricks Delta schema evolution",
    )

    second_response_file = write_llm_response(
        tmp_path,
        "second_ingest.json",
        ingest_response(
            content_type="correction",
            peer_links=[
                {
                    "path": AUTO_LOADER_SCHEMA_PATH,
                    "reason": "Model approved this peer.",
                }
            ],
            stale_candidates=[
                {
                    "path": STRUCTURED_SCHEMA_PATH,
                    "reason": "Model identified this stale candidate.",
                }
            ],
        ),
    )
    output = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(second_response_file),
        "ingest",
        "For Delta streaming sink schema evolution use mergeSchema=true",
    )

    assert "Stale candidates:" in output
    assert f"- {STRUCTURED_SCHEMA_PATH}:" in output
    store = JsonStore(tmp_path / "data")
    chunks = store.get_chunks_by_path(STRUCTURED_SCHEMA_PATH)
    assert len(chunks) == 2
    assert {chunk.status for chunk in chunks} == {"active"}


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
    assert payload["confidence"] == 0.0
    assert "Clarification needed:" in output
    assert "Allowed context:" not in output


def test_cli_map_prints_namespace_map_without_raw_chunk_content(monkeypatch, capsys, tmp_path):
    response_file = write_llm_response(tmp_path, "ingest.json", ingest_response())
    run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(response_file),
        "ingest",
        "Databricks raw note hidden from namespace map.",
    )
    store = JsonStore(tmp_path / "data")
    store.append_node_gold_aspect("WORK/DataArt", "Stable DataArt summary")

    output = run_cli(monkeypatch, capsys, tmp_path, "map", "--path", "WORK/DataArt", "--max-depth", "1")

    assert "Namespace map:" in output
    assert "WORK/DataArt" in output
    assert "gold: Stable DataArt summary" in output
    assert "subtree=1" in output
    assert "Databricks raw note hidden" not in output


def test_cli_map_json_is_model_readable(monkeypatch, capsys, tmp_path):
    response_file = write_llm_response(tmp_path, "ingest.json", ingest_response())
    run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(response_file),
        "ingest",
        "Databricks raw note hidden from namespace map.",
    )

    output = run_cli(monkeypatch, capsys, tmp_path, "map", "--json", "--path", "WORK/DataArt")

    payload = json.loads(output)
    paths = {node["path"] for node in payload["nodes"]}
    assert payload["root_path"] == "WORK/DataArt"
    assert STRUCTURED_SCHEMA_PATH in paths
    assert "Databricks raw note hidden" not in output


def test_cli_uses_configured_data_dir(monkeypatch, capsys, tmp_path):
    response_file = write_llm_response(
        tmp_path,
        "dbt_ingest.json",
        ingest_response(
            target_path="WORK/Stack/dbt",
            content_type="decision",
        ),
    )
    output = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--data-dir",
        "brain-data",
        "--llm-response-file",
        str(response_file),
        "ingest",
        "dbt ephemeral staging model",
    )

    assert "Target path: WORK/Stack/dbt" in output
    store = JsonStore(tmp_path / "brain-data")
    assert len(store.get_chunks_by_path("WORK/Stack/dbt")) == 1
    assert not (tmp_path / "data" / "chunks.json").exists()


def test_cli_optimize_marks_duplicates_and_reports_gold_file(monkeypatch, capsys, tmp_path):
    response_file = write_llm_response(tmp_path, "ingest.json", ingest_response())
    run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(response_file),
        "ingest",
        "Databricks Delta schema evolution",
    )
    run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(response_file),
        "ingest",
        "Databricks Delta schema evolution",
    )

    output = run_cli(monkeypatch, capsys, tmp_path, "optimize", STRUCTURED_SCHEMA_PATH)

    assert "1 exact duplicates marked stale" in output
    store = JsonStore(tmp_path / "data")
    chunks = store.get_chunks_by_path(STRUCTURED_SCHEMA_PATH)
    assert [chunk.status for chunk in chunks].count("stale") == 1


def test_cli_operation_dry_run_does_not_mutate_store(monkeypatch, capsys, tmp_path):
    operation_file = tmp_path / "operation.json"
    operation_file.write_text(
        json.dumps(
            {
                "operation": "append_chunk",
                "target_path": "WORK/Vertical/Node",
                "chunk": {"content": "Protocol dry-run note."},
            }
        ),
        encoding="utf-8",
    )

    output = run_cli(monkeypatch, capsys, tmp_path, "operation", "dry-run", str(operation_file))

    payload = json.loads(output)
    assert payload["status"] == "dry_run"
    assert payload["validation"]["valid"] is True
    store = JsonStore(tmp_path / "data")
    assert store.get_chunks_by_path("WORK/Vertical/Node") == []


def test_cli_operation_apply_writes_operation_batch(monkeypatch, capsys, tmp_path):
    operation_file = tmp_path / "operation_batch.json"
    operation_file.write_text(
        json.dumps(
            {
                "operations": [
                    {
                        "operation": "append_chunk",
                        "target_path": "WORK/Vertical/Node",
                        "chunk": {
                            "content": "Applied protocol note.",
                            "layer": "silver",
                            "content_type": "fact",
                        },
                    }
                ],
                "reasoning_summary": "Model emitted one storage operation.",
            }
        ),
        encoding="utf-8",
    )

    output = run_cli(monkeypatch, capsys, tmp_path, "operation", "apply", str(operation_file))

    payload = json.loads(output)
    assert payload["status"] == "applied"
    assert payload["validation"]["valid"] is True
    store = JsonStore(tmp_path / "data")
    chunks = store.get_chunks_by_path("WORK/Vertical/Node")
    assert len(chunks) == 1
    assert chunks[0].content == "Applied protocol note."


def test_cli_can_use_sqlite_storage_backend(monkeypatch, capsys, tmp_path):
    response_file = write_llm_response(tmp_path, "ingest.json", ingest_response())

    output = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--storage-backend",
        "sqlite",
        "--llm-response-file",
        str(response_file),
        "ingest",
        "Databricks SQLite-backed fact",
    )

    assert f"Target path: {STRUCTURED_SCHEMA_PATH}" in output
    assert (tmp_path / "data" / "vertical_brain.sqlite").exists()
    assert not (tmp_path / "data" / "chunks.json").exists()
    store = SQLiteStore(tmp_path / "data")
    chunks = store.get_chunks_by_path(STRUCTURED_SCHEMA_PATH)
    assert len(chunks) == 1
    assert chunks[0].content == "Databricks SQLite-backed fact"


def test_cli_search_finds_ingested_json_chunk(monkeypatch, capsys, tmp_path):
    response_file = write_llm_response(tmp_path, "ingest.json", ingest_response())
    run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(response_file),
        "ingest",
        "Databricks Delta schema evolution uses mergeSchema.",
    )

    output = run_cli(monkeypatch, capsys, tmp_path, "search", "--path", "WORK/DataArt", "mergeSchema")

    assert "Search results:" in output
    assert f"[{STRUCTURED_SCHEMA_PATH}][chunk/silver/fact]" in output
    assert "mergeSchema" in output


def test_cli_search_uses_sqlite_fts_backend(monkeypatch, capsys, tmp_path):
    response_file = write_llm_response(tmp_path, "ingest.json", ingest_response())
    run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--storage-backend",
        "sqlite",
        "--llm-response-file",
        str(response_file),
        "ingest",
        "Databricks SQLite FTS search note.",
    )

    output = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--storage-backend",
        "sqlite",
        "search",
        "FTS",
    )

    assert "Search results:" in output
    assert f"[{STRUCTURED_SCHEMA_PATH}][chunk/silver/fact]" in output
    assert "SQLite FTS search note" in output


def test_cli_context_search_opens_locked_context_from_candidate_paths(monkeypatch, capsys, tmp_path):
    structured_response_file = write_llm_response(tmp_path, "structured.json", ingest_response())
    run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(structured_response_file),
        "ingest",
        "needle needle needle locked target fact",
    )
    auto_loader_response_file = write_llm_response(
        tmp_path,
        "auto_loader.json",
        ingest_response(target_path=AUTO_LOADER_SCHEMA_PATH),
    )
    run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(auto_loader_response_file),
        "ingest",
        "needle omitted candidate fact",
    )

    output = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "context",
        "search",
        "--path",
        "WORK/DataArt",
        "--context-limit",
        "1",
        "needle",
    )

    assert "Candidate handles:" in output
    assert "Locked contexts:" in output
    assert f"Context: {STRUCTURED_SCHEMA_PATH}" in output
    assert "needle needle needle locked target fact" in output
    assert "needle omitted candidate fact" not in output
    assert "Omitted candidate paths: 1" in output


def test_cli_context_search_json_omits_raw_search_snippets(monkeypatch, capsys, tmp_path):
    response_file = write_llm_response(tmp_path, "ingest.json", ingest_response())
    run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(response_file),
        "ingest",
        "schema evolution locked target fact",
    )

    output = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "context",
        "search",
        "--json",
        "schema evolution",
    )

    payload = json.loads(output)
    assert payload["candidate_handles"][0]["path"] == STRUCTURED_SCHEMA_PATH
    assert "snippet" not in payload["candidate_handles"][0]
    assert payload["locked_contexts"][0]["target_path"] == STRUCTURED_SCHEMA_PATH
    assert "schema evolution locked target fact" in payload["locked_contexts"][0]["items"][0]["content"]


def test_cli_context_expand_opens_link_target_as_locked_context(monkeypatch, capsys, tmp_path):
    auto_loader_response_file = write_llm_response(
        tmp_path,
        "auto_loader.json",
        ingest_response(target_path=AUTO_LOADER_SCHEMA_PATH),
    )
    run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(auto_loader_response_file),
        "ingest",
        "Auto Loader expanded target context.",
    )
    structured_response_file = write_llm_response(
        tmp_path,
        "structured.json",
        ingest_response(
            peer_links=[
                {
                    "path": AUTO_LOADER_SCHEMA_PATH,
                    "reason": "Model approved this peer.",
                }
            ]
        ),
    )
    run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(structured_response_file),
        "ingest",
        "Structured Streaming source context should stay locked.",
    )
    store = JsonStore(tmp_path / "data")
    link_id = store.get_peer_links(STRUCTURED_SCHEMA_PATH)[0].id

    output = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "context",
        "expand",
        link_id,
        "--from-path",
        STRUCTURED_SCHEMA_PATH,
    )

    assert f"Expanded link: {link_id}" in output
    assert f"Expanded path: {AUTO_LOADER_SCHEMA_PATH}" in output
    assert "Auto Loader expanded target context." in output
    assert "Structured Streaming source context should stay locked." not in output


def test_cli_context_expand_json_is_model_readable(monkeypatch, capsys, tmp_path):
    auto_loader_response_file = write_llm_response(
        tmp_path,
        "auto_loader.json",
        ingest_response(target_path=AUTO_LOADER_SCHEMA_PATH),
    )
    run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(auto_loader_response_file),
        "ingest",
        "Auto Loader JSON expanded context.",
    )
    structured_response_file = write_llm_response(
        tmp_path,
        "structured.json",
        ingest_response(
            peer_links=[
                {
                    "path": AUTO_LOADER_SCHEMA_PATH,
                    "reason": "Model approved this peer.",
                }
            ]
        ),
    )
    run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "--llm-response-file",
        str(structured_response_file),
        "ingest",
        "Structured Streaming linked source context.",
    )
    store = JsonStore(tmp_path / "data")
    link_id = store.get_peer_links(STRUCTURED_SCHEMA_PATH)[0].id

    output = run_cli(
        monkeypatch,
        capsys,
        tmp_path,
        "context",
        "expand",
        link_id,
        "--from-path",
        STRUCTURED_SCHEMA_PATH,
        "--json",
    )

    payload = json.loads(output)
    assert payload["link_id"] == link_id
    assert payload["expanded_path"] == AUTO_LOADER_SCHEMA_PATH
    assert payload["locked_context"]["target_path"] == AUTO_LOADER_SCHEMA_PATH
    assert payload["locked_context"]["items"][0]["content"] == "Auto Loader JSON expanded context."


def test_cli_operation_dry_run_reports_schema_errors(monkeypatch, capsys, tmp_path):
    operation_file = tmp_path / "invalid_operation.json"
    operation_file.write_text(
        json.dumps(
            {
                "operation": "append_chunk",
                "target_path": "WORK/Vertical/Node",
                "chunk": {
                    "content": "Valid content.",
                    "made_up": True,
                },
            }
        ),
        encoding="utf-8",
    )

    output = run_cli(monkeypatch, capsys, tmp_path, "operation", "dry-run", str(operation_file))

    payload = json.loads(output)
    assert payload["status"] == "invalid"
    assert payload["validation"]["valid"] is False
    assert any(issue["path"] == "$.chunk.made_up" for issue in payload["validation"]["issues"])
    store = JsonStore(tmp_path / "data")
    assert store.get_chunks_by_path("WORK/Vertical/Node") == []


def test_cli_operation_apply_rejects_schema_errors(monkeypatch, capsys, tmp_path):
    operation_file = tmp_path / "invalid_operation.json"
    operation_file.write_text(
        json.dumps(
            {
                "operation": "append_chunk",
                "target_path": "WORK/Vertical/Node",
                "chunk": {
                    "content": "Valid content.",
                    "made_up": True,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="made_up"):
        run_cli(monkeypatch, capsys, tmp_path, "operation", "apply", str(operation_file))

    store = JsonStore(tmp_path / "data")
    assert store.get_chunks_by_path("WORK/Vertical/Node") == []
