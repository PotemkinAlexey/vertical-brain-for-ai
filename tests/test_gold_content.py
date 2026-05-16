import json
from pathlib import Path

from vertical_brain.core.gold import parse_gold_content
from vertical_brain.core.router import StorageModel, ContextSessionConfig

MODEL_FILE = Path(__file__).resolve().parents[1] / "data" / "namespaces" / "model.json"


def test_parse_gold_content_handles_plain_text():
    assert parse_gold_content("Delta migration | AutoLoader streaming") == [
        "Delta migration",
        "AutoLoader streaming",
    ]


def test_parse_gold_content_handles_structured_json():
    content = json.dumps({
        "aspects": ["Delta migration", "AutoLoader streaming"],
        "last_updated": "2026-01-01",
    })
    assert parse_gold_content(content) == ["Delta migration", "AutoLoader streaming"]


def test_parse_gold_content_handles_gold_document_json():
    content = json.dumps({
        "facts": [
            {"content": "Delta migration requires cluster policy review."},
            {"content": "AutoLoader schema drift needs explicit handling."},
        ],
        "entities": ["Delta", "AutoLoader"],
        "rules": [],
    })

    assert parse_gold_content(content) == [
        "Delta migration requires cluster policy review.",
        "AutoLoader schema drift needs explicit handling.",
    ]


def test_parse_gold_content_handles_empty():
    assert parse_gold_content("") == []
    assert parse_gold_content("   ") == []


def test_context_session_config_loads_from_model_json():
    model = StorageModel.load(MODEL_FILE)
    cfg = model.context_session_config
    assert isinstance(cfg, ContextSessionConfig)
    assert cfg.default_search_limit == 10
    assert cfg.default_context_limit == 3
    assert cfg.default_items_per_context == 6
    assert cfg.default_max_context_items == 12
    assert cfg.default_link_expansion == "handles_only"


def test_context_session_config_falls_back_to_defaults():
    model = StorageModel({"context_policy": {}})
    cfg = model.context_session_config
    assert cfg.default_search_limit == 10
    assert cfg.default_link_expansion == "handles_only"
