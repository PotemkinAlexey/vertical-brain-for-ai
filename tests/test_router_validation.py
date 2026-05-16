import json
from pathlib import Path

import pytest

from vertical_brain.core.router import LLMRouter, StorageModel

MODEL_FILE = Path(__file__).resolve().parents[1] / "data" / "namespaces" / "model.json"


class StubLLM:
    def __init__(self, payload):
        self._payload = payload

    def complete(self, prompt):  # noqa: ARG002
        return json.dumps(self._payload)


def _router(payload):
    return LLMRouter(StubLLM(payload), StorageModel.load(MODEL_FILE))


def _valid_route():
    return {
        "target_path": "WORK/DataArt/Databricks",
        "content_type": "fact",
        "layer": "silver",
        "action": "append_silver",
        "peer_links": [],
        "stale_candidates": [],
        "confidence": 0.9,
        "reasoning_summary": "ok",
    }


def _valid_query():
    return {
        "target_path": "WORK/DataArt/Databricks",
        "allowed_context": {
            "include_ancestors": True,
            "include_peer_links": True,
            "exclude_other_branches": True,
        },
        "query_type": "lookup",
        "confidence": 0.9,
        "reasoning_summary": "ok",
    }


def test_valid_route_decision_passes():
    decision = _router(_valid_route()).route_ingest("text")
    assert decision.target_path == "WORK/DataArt/Databricks"


def test_route_decision_rejects_missing_required_field():
    payload = _valid_route()
    del payload["layer"]
    with pytest.raises(ValueError):
        _router(payload).route_ingest("text")


def test_route_decision_rejects_invalid_layer_enum():
    payload = _valid_route()
    payload["layer"] = "platinum"
    with pytest.raises(ValueError):
        _router(payload).route_ingest("text")


def test_route_decision_rejects_confidence_out_of_range():
    payload = _valid_route()
    payload["confidence"] = 1.5
    with pytest.raises(ValueError):
        _router(payload).route_ingest("text")


def test_query_route_decision_rejects_invalid_query_type():
    payload = _valid_query()
    payload["query_type"] = "wild_guess"
    with pytest.raises(ValueError):
        _router(payload).route_query("question?")


def test_query_route_decision_rejects_malformed_allowed_context():
    payload = _valid_query()
    payload["allowed_context"]["unexpected_field"] = True
    with pytest.raises(ValueError):
        _router(payload).route_query("question?")
