from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from vertical_brain.core.models import (
    AllowedContext,
    PeerLinkCandidate,
    QueryRouteDecision,
    RouteDecision,
    StaleCandidate,
)


class LLMProvider(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class StorageModel:
    """Storage and contract format metadata loaded from model.json."""

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    @classmethod
    def load(cls, path: str | Path) -> StorageModel:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def namespace_path_shape(self) -> list[str]:
        return list(self.payload["namespace_path_shape"])

    @property
    def min_compaction_path_parts(self) -> int:
        return len(self.namespace_path_shape)

    @property
    def clarification_threshold(self) -> float:
        return float(self.payload["routing_contract"]["clarification_threshold"])

    @property
    def route_decision_fields(self) -> list[str]:
        return list(self.payload["routing_contract"]["route_decision_fields"])

    @property
    def query_route_decision_fields(self) -> list[str]:
        return list(self.payload["routing_contract"]["query_route_decision_fields"])

    @property
    def storage_operation_payload_schema(self) -> dict[str, Any]:
        return dict(self.payload["storage_operation_contract"]["payload_schema"])


class LLMRouter:
    """Routes inputs by asking a model for strict JSON decisions."""

    def __init__(self, llm: LLMProvider, storage_model: StorageModel):
        self.llm = llm
        self.storage_model = storage_model

    def route_ingest(self, text: str, known_namespaces: list[str] | None = None) -> RouteDecision:
        payload = self._complete_json(
            self._build_prompt(
                mode="ingest",
                text=text,
                known_namespaces=known_namespaces or [],
                required_fields=self.storage_model.route_decision_fields,
            )
        )
        return self._route_decision_from_payload(payload)

    def route_query(self, question: str, known_namespaces: list[str] | None = None) -> QueryRouteDecision:
        payload = self._complete_json(
            self._build_prompt(
                mode="ask",
                text=question,
                known_namespaces=known_namespaces or [],
                required_fields=self.storage_model.query_route_decision_fields,
            )
        )
        return self._query_route_decision_from_payload(payload)

    def requires_clarification(self, decision: RouteDecision | QueryRouteDecision) -> bool:
        action = getattr(decision, "action", None)
        return action == "ask_clarification" or decision.confidence < self.storage_model.clarification_threshold

    def _build_prompt(
        self,
        mode: str,
        text: str,
        known_namespaces: list[str],
        required_fields: list[str],
    ) -> str:
        payload = {
            "mode": mode,
            "namespace_path_shape": self.storage_model.namespace_path_shape,
            "known_namespaces": known_namespaces,
            "required_fields": required_fields,
            "input": text,
            "instruction": "Return one strict JSON object and no prose.",
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _complete_json(self, prompt: str) -> dict[str, Any]:
        raw = self.llm.complete(prompt)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM router returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("LLM router must return a JSON object")
        return payload

    def _route_decision_from_payload(self, payload: dict[str, Any]) -> RouteDecision:
        self._require_fields(payload, self.storage_model.route_decision_fields)
        return RouteDecision(
            target_path=payload["target_path"],
            content_type=payload["content_type"],
            layer=payload["layer"],
            action=payload["action"],
            peer_links=[
                PeerLinkCandidate(path=peer["path"], reason=peer["reason"])
                for peer in payload.get("peer_links", [])
            ],
            stale_candidates=[
                StaleCandidate(path=candidate["path"], reason=candidate["reason"])
                for candidate in payload.get("stale_candidates", [])
            ],
            confidence=float(payload["confidence"]),
            reasoning_summary=payload["reasoning_summary"],
        )

    def _query_route_decision_from_payload(self, payload: dict[str, Any]) -> QueryRouteDecision:
        self._require_fields(payload, self.storage_model.query_route_decision_fields)
        allowed_context = payload["allowed_context"]
        return QueryRouteDecision(
            target_path=payload["target_path"],
            allowed_context=AllowedContext(
                include_ancestors=allowed_context["include_ancestors"],
                include_peer_links=allowed_context["include_peer_links"],
                exclude_other_branches=allowed_context["exclude_other_branches"],
            ),
            query_type=payload["query_type"],
            confidence=float(payload["confidence"]),
            reasoning_summary=payload["reasoning_summary"],
        )

    def _require_fields(self, payload: dict[str, Any], fields: list[str]) -> None:
        missing_fields = [field for field in fields if field not in payload]
        if missing_fields:
            raise ValueError(f"LLM router response missing fields: {', '.join(missing_fields)}")
