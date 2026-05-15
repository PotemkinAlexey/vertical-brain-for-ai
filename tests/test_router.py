import json

from vertical_brain.core.router import MockRouter

STRUCTURED_SCHEMA_PATH = "WORK/DataArt/Databricks/Certification/StructuredStreaming/SchemaEvolution"
AUTO_LOADER_SCHEMA_PATH = "WORK/DataArt/Databricks/Certification/AutoLoader/SchemaEvolution"


def test_databricks_route():
    router = MockRouter()
    decision = router.route_ingest("Databricks Delta schema evolution")
    assert decision.target_path == STRUCTURED_SCHEMA_PATH


def test_dbt_route():
    router = MockRouter()
    decision = router.route_ingest("dbt ephemeral staging model")
    assert decision.target_path == "WORK/Stack/dbt"


def test_unknown_route():
    router = MockRouter()
    decision = router.route_ingest("random note")
    assert decision.target_path == "INBOX/Unclassified"
    assert decision.action == "ask_clarification"


def test_query_route_uses_ask_contract():
    router = MockRouter()
    decision = router.route_query("How does Databricks schema evolution work?")

    assert decision.target_path == STRUCTURED_SCHEMA_PATH
    assert decision.query_type == "explanation"
    assert decision.allowed_context.include_ancestors is True
    assert decision.allowed_context.include_peer_links is True
    assert decision.allowed_context.exclude_other_branches is True


def test_route_decisions_serialize_to_strict_json():
    router = MockRouter()
    ingest_decision = router.route_ingest("Databricks Auto Loader schema evolution")
    query_decision = router.route_query("How does Databricks schema evolution work?")

    ingest_payload = json.loads(ingest_decision.to_json())
    query_payload = json.loads(query_decision.to_json())

    assert ingest_payload["target_path"] == AUTO_LOADER_SCHEMA_PATH
    assert ingest_payload["peer_links"][0]["path"] == STRUCTURED_SCHEMA_PATH
    assert query_payload["allowed_context"]["exclude_other_branches"] is True


def test_delta_streaming_schema_evolution_adds_peer_link_and_stale_candidate():
    router = MockRouter()
    decision = router.route_ingest("For Delta streaming sink schema evolution use mergeSchema=true")

    assert decision.target_path == STRUCTURED_SCHEMA_PATH
    assert decision.content_type == "correction"
    assert decision.peer_links[0].path == AUTO_LOADER_SCHEMA_PATH
    assert decision.stale_candidates[0].path == STRUCTURED_SCHEMA_PATH
