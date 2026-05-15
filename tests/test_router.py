from vertical_brain.core.router import MockRouter


def test_databricks_route():
    router = MockRouter()
    decision = router.route_ingest("Databricks Delta schema evolution")
    assert decision.target_path == "WORK/DataArt/Databricks"


def test_dbt_route():
    router = MockRouter()
    decision = router.route_ingest("dbt ephemeral staging model")
    assert decision.target_path == "WORK/Stack/dbt"


def test_unknown_route():
    router = MockRouter()
    decision = router.route_ingest("random note")
    assert decision.target_path == "INBOX/Unclassified"


def test_query_route_uses_ask_contract():
    router = MockRouter()
    decision = router.route_query("How does Databricks schema evolution work?")

    assert decision.target_path == "WORK/DataArt/Databricks"
    assert decision.query_type == "explanation"
    assert decision.allowed_context.include_ancestors is True
    assert decision.allowed_context.include_peer_links is True
    assert decision.allowed_context.exclude_other_branches is True
