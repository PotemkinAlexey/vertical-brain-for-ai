import json
from pathlib import Path

from vertical_brain.core.gold import gold_embed_text, parse_gold_content
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


# ── gold_embed_text ────────────────────────────────────────────────────────────

def test_gold_embed_text_strips_json_noise_from_v2_format():
    """v2 JSON with UUIDs and timestamps should produce clean joined text."""
    content = json.dumps({
        "aspects": [
            {"id": "550e8400-e29b-41d4-a716-446655440000", "text": "Delta Lake Z-ordering reduces scan range", "updated_at": "2026-05-17T10:00:00+00:00"},
            {"id": "7b8c12de-cafe-beef-dead-000000000001", "text": "AutoLoader detects schema drift automatically", "updated_at": "2026-05-17T11:00:00+00:00"},
        ]
    })
    result = gold_embed_text(content)
    assert result == "Delta Lake Z-ordering reduces scan range | AutoLoader detects schema drift automatically"
    # UUIDs and timestamps must not appear in the embed text
    assert "550e8400" not in result
    assert "2026-05-17" not in result


def test_gold_embed_text_handles_plain_text_passthrough():
    """Plain text Gold (legacy format) should pass through unchanged."""
    content = "Delta migration | AutoLoader streaming"
    assert gold_embed_text(content) == "Delta migration | AutoLoader streaming"


def test_gold_embed_text_handles_single_aspect():
    content = json.dumps({
        "aspects": [
            {"id": "abc", "text": "Databricks Delta Lake is the storage layer", "updated_at": "2026-01-01T00:00:00+00:00"},
        ]
    })
    assert gold_embed_text(content) == "Databricks Delta Lake is the storage layer"


def test_gold_embed_text_falls_back_to_raw_content_on_empty_parse():
    """If no aspects can be parsed, return the raw content as fallback."""
    raw = "some unstructured text that is not JSON"
    assert gold_embed_text(raw) == raw


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
