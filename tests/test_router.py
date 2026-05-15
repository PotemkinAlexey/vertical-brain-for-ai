import json
from pathlib import Path

import pytest

from vertical_brain.core.router import LLMRouter, StorageModel
from vertical_brain.llm.mock_llm import MockLLM

STRUCTURED_SCHEMA_PATH = "WORK/DataArt/Databricks/Certification/StructuredStreaming/SchemaEvolution"
AUTO_LOADER_SCHEMA_PATH = "WORK/DataArt/Databricks/Certification/AutoLoader/SchemaEvolution"
MODEL_FILE = Path(__file__).resolve().parents[1] / "data" / "namespaces" / "model.json"


def build_router(*responses: dict) -> LLMRouter:
    return LLMRouter(
        MockLLM([json.dumps(response) for response in responses]),
        StorageModel.load(MODEL_FILE),
    )


def test_storage_model_file_is_contract_not_routing_data():
    payload = json.loads(MODEL_FILE.read_text(encoding="utf-8"))

    assert "routing_contract" in payload
    assert "routing" not in payload
    assert "rules" not in payload


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


def test_ingest_route_comes_from_llm_response():
    router = build_router(ingest_response())

    decision = router.route_ingest("Databricks Delta schema evolution")

    assert decision.target_path == STRUCTURED_SCHEMA_PATH
    assert decision.layer == "silver"
    assert decision.action == "append_and_optimize"


def test_query_route_comes_from_llm_response():
    router = build_router(query_response())

    decision = router.route_query("How does Databricks schema evolution work?")

    assert decision.target_path == STRUCTURED_SCHEMA_PATH
    assert decision.query_type == "explanation"
    assert decision.allowed_context.include_ancestors is True
    assert decision.allowed_context.include_peer_links is True
    assert decision.allowed_context.exclude_other_branches is True


def test_route_decisions_serialize_to_strict_json():
    router = build_router(
        ingest_response(
            target_path=AUTO_LOADER_SCHEMA_PATH,
            peer_links=[
                {
                    "path": STRUCTURED_SCHEMA_PATH,
                    "reason": "Model approved this comparison peer.",
                }
            ],
        ),
        query_response(),
    )

    ingest_decision = router.route_ingest("Databricks Auto Loader schema evolution")
    query_decision = router.route_query("How does Databricks schema evolution work?")

    ingest_payload = json.loads(ingest_decision.to_json())
    query_payload = json.loads(query_decision.to_json())

    assert ingest_payload["target_path"] == AUTO_LOADER_SCHEMA_PATH
    assert ingest_payload["peer_links"][0]["path"] == STRUCTURED_SCHEMA_PATH
    assert query_payload["allowed_context"]["exclude_other_branches"] is True


def test_ingest_route_preserves_peer_links_and_stale_candidates_from_model():
    router = build_router(
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
        )
    )

    decision = router.route_ingest("For Delta streaming sink schema evolution use mergeSchema=true")

    assert decision.target_path == STRUCTURED_SCHEMA_PATH
    assert decision.content_type == "correction"
    assert decision.peer_links[0].path == AUTO_LOADER_SCHEMA_PATH
    assert decision.stale_candidates[0].path == STRUCTURED_SCHEMA_PATH


def test_unknown_route_defaults_to_clarification_without_provider_data():
    router = LLMRouter(MockLLM(), StorageModel.load(MODEL_FILE))

    decision = router.route_ingest("random note")

    assert decision.target_path == "INBOX/Unclassified"
    assert decision.action == "ask_clarification"
    assert router.requires_clarification(decision) is True


def test_router_rejects_missing_required_fields():
    router = build_router({"target_path": "WORK/Incomplete"})

    with pytest.raises(ValueError, match="missing fields"):
        router.route_ingest("anything")
