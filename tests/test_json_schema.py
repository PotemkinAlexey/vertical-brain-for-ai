import json
from pathlib import Path

from vertical_brain.core.json_schema import validate_json_schema
from vertical_brain.core.router import StorageModel


MODEL_FILE = Path(__file__).resolve().parents[1] / "data" / "namespaces" / "model.json"


def load_operation_schema():
    return StorageModel.load(MODEL_FILE).storage_operation_payload_schema


def test_storage_operation_schema_accepts_single_operation():
    payload = {
        "operation": "append_chunk",
        "target_path": "WORK/Vertical/Node",
        "chunk": {
            "content": "Schema-valid note.",
            "layer": "silver",
            "content_type": "fact",
            "lineage": ["source-a"],
        },
    }

    validation = validate_json_schema(payload, load_operation_schema())

    assert validation.valid is True
    assert validation.issues == []


def test_storage_operation_schema_accepts_batch():
    payload = {
        "operations": [
            {
                "operation": "create_node",
                "target_path": "WORK/Vertical/Node",
            }
        ],
        "reasoning_summary": "Create a namespace.",
    }

    validation = validate_json_schema(payload, load_operation_schema())

    assert validation.valid is True


def test_storage_operation_schema_rejects_unknown_fields_and_bad_enums():
    payload = {
        "operation": "append_magic",
        "target_path": "WORK/Vertical/Node",
        "unexpected": True,
    }

    validation = validate_json_schema(payload, load_operation_schema())

    messages = [issue.message for issue in validation.issues]
    assert validation.valid is False
    assert "must match exactly one allowed schema" in messages
    assert "must be one of: create_node, append_chunk, create_link, mark_stale, supersede_chunk, append_gold_aspect" in messages
    assert "is not allowed" in messages


def test_storage_operation_contract_in_model_json_has_no_domain_data():
    payload = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
    serialized = json.dumps(payload["storage_operation_contract"], ensure_ascii=False).lower()

    assert "databricks" not in serialized
    assert "dataart" not in serialized
