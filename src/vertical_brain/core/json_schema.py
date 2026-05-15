from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vertical_brain.core.models import ValidationIssue, ValidationResult


def validate_json_schema(payload: object, schema: Mapping[str, Any]) -> ValidationResult:
    issues: list[ValidationIssue] = []
    _validate(payload, schema, root_schema=schema, path="$", issues=issues)
    return ValidationResult(valid=not issues, issues=issues)


def format_json_schema_errors(validation: ValidationResult) -> str:
    return "; ".join(f"{issue.path}: {issue.message}" for issue in validation.issues)


def _validate(
    payload: object,
    schema: Mapping[str, Any],
    *,
    root_schema: Mapping[str, Any],
    path: str,
    issues: list[ValidationIssue],
) -> None:
    if "$ref" in schema:
        _validate(payload, _resolve_ref(schema["$ref"], root_schema), root_schema=root_schema, path=path, issues=issues)
        return

    if "oneOf" in schema:
        _validate_one_of(payload, schema["oneOf"], root_schema=root_schema, path=path, issues=issues)
        return

    if "enum" in schema and payload not in schema["enum"]:
        allowed = ", ".join(str(item) for item in schema["enum"])
        issues.append(ValidationIssue(path=path, message=f"must be one of: {allowed}"))
        return

    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(payload, expected_type):
        issues.append(ValidationIssue(path=path, message=f"must be {expected_type}"))
        return

    if expected_type == "object" or isinstance(payload, Mapping):
        _validate_object(payload, schema, root_schema=root_schema, path=path, issues=issues)
    elif expected_type == "array" or isinstance(payload, list):
        _validate_array(payload, schema, root_schema=root_schema, path=path, issues=issues)
    elif expected_type == "string" and isinstance(payload, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(payload) < min_length:
            issues.append(ValidationIssue(path=path, message=f"length must be at least {min_length}"))
    elif expected_type in {"number", "integer"} and isinstance(payload, (int, float)) and not isinstance(payload, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and payload < minimum:
            issues.append(ValidationIssue(path=path, message=f"must be >= {minimum}"))
        if isinstance(maximum, (int, float)) and payload > maximum:
            issues.append(ValidationIssue(path=path, message=f"must be <= {maximum}"))


def _validate_one_of(
    payload: object,
    schemas: object,
    *,
    root_schema: Mapping[str, Any],
    path: str,
    issues: list[ValidationIssue],
) -> None:
    if not isinstance(schemas, list):
        issues.append(ValidationIssue(path=path, message="schema oneOf must be a list"))
        return

    matched_count = 0
    first_errors: list[ValidationIssue] = []
    for candidate_schema in schemas:
        if not isinstance(candidate_schema, Mapping):
            continue
        candidate_issues: list[ValidationIssue] = []
        _validate(payload, candidate_schema, root_schema=root_schema, path=path, issues=candidate_issues)
        if not candidate_issues:
            matched_count += 1
        elif not first_errors:
            first_errors = candidate_issues

    if matched_count == 1:
        return
    if matched_count == 0:
        issues.append(ValidationIssue(path=path, message="must match exactly one allowed schema"))
        issues.extend(first_errors[:3])
        return
    issues.append(ValidationIssue(path=path, message="matches more than one allowed schema"))


def _validate_object(
    payload: object,
    schema: Mapping[str, Any],
    *,
    root_schema: Mapping[str, Any],
    path: str,
    issues: list[ValidationIssue],
) -> None:
    if not isinstance(payload, Mapping):
        return

    required = schema.get("required", [])
    if isinstance(required, list):
        for field in required:
            if isinstance(field, str) and field not in payload:
                issues.append(ValidationIssue(path=f"{path}.{field}", message="is required"))

    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        return

    if schema.get("additionalProperties") is False:
        for field in sorted(set(payload) - set(properties)):
            issues.append(ValidationIssue(path=f"{path}.{field}", message="is not allowed"))

    for field, value in payload.items():
        field_schema = properties.get(field)
        if isinstance(field_schema, Mapping):
            _validate(value, field_schema, root_schema=root_schema, path=f"{path}.{field}", issues=issues)


def _validate_array(
    payload: object,
    schema: Mapping[str, Any],
    *,
    root_schema: Mapping[str, Any],
    path: str,
    issues: list[ValidationIssue],
) -> None:
    if not isinstance(payload, list):
        return

    min_items = schema.get("minItems")
    if isinstance(min_items, int) and len(payload) < min_items:
        issues.append(ValidationIssue(path=path, message=f"must contain at least {min_items} items"))

    item_schema = schema.get("items")
    if not isinstance(item_schema, Mapping):
        return
    for index, item in enumerate(payload):
        _validate(item, item_schema, root_schema=root_schema, path=f"{path}[{index}]", issues=issues)


def _matches_type(payload: object, expected_type: object) -> bool:
    if expected_type == "object":
        return isinstance(payload, Mapping)
    if expected_type == "array":
        return isinstance(payload, list)
    if expected_type == "string":
        return isinstance(payload, str)
    if expected_type == "number":
        return isinstance(payload, (int, float)) and not isinstance(payload, bool)
    if expected_type == "integer":
        return isinstance(payload, int) and not isinstance(payload, bool)
    if expected_type == "boolean":
        return isinstance(payload, bool)
    if expected_type == "null":
        return payload is None
    return True


def _resolve_ref(ref: object, root_schema: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        raise ValueError(f"Unsupported JSON Schema ref: {ref}")

    target: object = root_schema
    for part in ref[2:].split("/"):
        if not isinstance(target, Mapping) or part not in target:
            raise ValueError(f"Unresolved JSON Schema ref: {ref}")
        target = target[part]
    if not isinstance(target, Mapping):
        raise ValueError(f"JSON Schema ref does not point to an object: {ref}")
    return target
