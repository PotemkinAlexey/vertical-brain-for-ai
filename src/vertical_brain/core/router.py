from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vertical_brain.core.models import (
    Action,
    AllowedContext,
    ContentType,
    Layer,
    PeerLinkCandidate,
    QueryRouteDecision,
    QueryType,
    RouteDecision,
    StaleCandidate,
)


class NamespaceModel:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    @classmethod
    def load(cls, path: str | Path) -> NamespaceModel:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def routing(self) -> dict[str, Any]:
        return self.payload.get("routing", {})

    @property
    def rules(self) -> list[dict[str, Any]]:
        return self.routing.get("rules", [])

    @property
    def default_route(self) -> dict[str, Any]:
        return self.routing["default_route"]

    @property
    def content_type_rules(self) -> list[dict[str, Any]]:
        return self.routing.get("content_type_rules", [])

    @property
    def query_type_rules(self) -> list[dict[str, Any]]:
        return self.routing.get("query_type_rules", [])

    @property
    def clarification_threshold(self) -> float:
        return float(self.routing["clarification_threshold"])

    @property
    def context_policy(self) -> dict[str, bool]:
        return self.payload.get("context_policy", {})

    @property
    def optimizer(self) -> dict[str, Any]:
        return self.payload.get("optimizer", {})


class ModelRouter:
    """Routes inputs by executing the namespace model, not by owning domain knowledge."""

    def __init__(self, model: NamespaceModel):
        self.model = model

    def route_ingest(self, text: str) -> RouteDecision:
        lowered = text.lower()
        for rule in self.model.rules:
            if self._matches(rule.get("match", {}), lowered):
                return self._route_decision_from_rule(rule, lowered)
        return self._route_decision_from_rule(self.model.default_route, lowered)

    def route_query(self, question: str) -> QueryRouteDecision:
        ingest_like_decision = self.route_ingest(question)
        context_policy = self.model.context_policy
        return QueryRouteDecision(
            target_path=ingest_like_decision.target_path,
            allowed_context=AllowedContext(
                include_ancestors=context_policy.get("include_ancestors", True),
                include_peer_links=context_policy.get("include_peer_links", True),
                exclude_other_branches=context_policy.get("exclude_other_branches", True),
            ),
            query_type=self._infer_query_type(question.lower()),
            confidence=ingest_like_decision.confidence,
            reasoning_summary=ingest_like_decision.reasoning_summary,
        )

    def requires_clarification(self, decision: RouteDecision | QueryRouteDecision) -> bool:
        action = getattr(decision, "action", None)
        return action == "ask_clarification" or decision.confidence < self.model.clarification_threshold

    def _route_decision_from_rule(self, rule: dict[str, Any], lowered: str) -> RouteDecision:
        return RouteDecision(
            target_path=rule["target_path"],
            content_type=self._content_type_for(rule, lowered),
            layer=self._layer(rule),
            action=self._action(rule),
            peer_links=[
                PeerLinkCandidate(path=peer["path"], reason=peer["reason"])
                for peer in rule.get("peer_links", [])
            ],
            stale_candidates=[
                StaleCandidate(path=candidate["path"], reason=candidate["reason"])
                for candidate in rule.get("stale_candidates", [])
            ],
            confidence=float(rule["confidence"]),
            reasoning_summary=rule["reasoning_summary"],
        )

    def _content_type_for(self, rule: dict[str, Any], lowered: str) -> ContentType:
        if "content_type" in rule:
            return rule["content_type"]
        for content_type_rule in self.model.content_type_rules:
            if self._matches(content_type_rule.get("match", {}), lowered):
                return content_type_rule["content_type"]
        return "fact"

    def _infer_query_type(self, lowered: str) -> QueryType:
        for query_type_rule in self.model.query_type_rules:
            if self._matches(query_type_rule.get("match", {}), lowered):
                return query_type_rule["query_type"]
        return "unknown"

    def _matches(self, match: dict[str, Any], lowered: str) -> bool:
        all_terms = [term.lower() for term in match.get("all", [])]
        any_terms = [term.lower() for term in match.get("any", [])]
        none_terms = [term.lower() for term in match.get("none", [])]
        starts_with = [term.lower() for term in match.get("starts_with", [])]
        ends_with = [term.lower() for term in match.get("ends_with", [])]

        if all_terms and not all(term in lowered for term in all_terms):
            return False
        if any_terms and not any(term in lowered for term in any_terms):
            return False
        if none_terms and any(term in lowered for term in none_terms):
            return False
        if starts_with and not any(lowered.startswith(term) for term in starts_with):
            return False
        if ends_with and not any(lowered.endswith(term) for term in ends_with):
            return False
        return any([all_terms, any_terms, none_terms, starts_with, ends_with])

    def _layer(self, rule: dict[str, Any]) -> Layer:
        return rule["layer"]

    def _action(self, rule: dict[str, Any]) -> Action:
        return rule["action"]
