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
