from __future__ import annotations

from vertical_brain.core.models import (
    AllowedContext,
    ContentType,
    PeerLinkCandidate,
    QueryRouteDecision,
    QueryType,
    RouteDecision,
    StaleCandidate,
)


class MockRouter:
    """Simple deterministic router for MVP v0.1.

    This should later be replaced by an LLM router that returns the same
    RouteDecision contract.
    """

    def route_ingest(self, text: str) -> RouteDecision:
        lowered = text.lower()

        if self._matches_databricks(lowered):
            return self._route_databricks(text)

        if "dbt" in lowered:
            return RouteDecision(
                target_path="WORK/Stack/dbt",
                content_type="decision",
                layer="silver",
                action="append_and_optimize",
                confidence=0.8,
                reasoning_summary="Matched dbt keyword.",
            )

        if "trading" in lowered or "bot" in lowered or "hedge" in lowered:
            return RouteDecision(
                target_path="TRADING",
                content_type="note",
                layer="bronze",
                action="append_bronze",
                confidence=0.7,
                reasoning_summary="Matched trading-related keyword.",
            )

        return RouteDecision(
            target_path="INBOX/Unclassified",
            content_type="note",
            layer="bronze",
            action="ask_clarification",
            confidence=0.4,
            reasoning_summary="No strong route found.",
        )

    def route_query(self, question: str) -> QueryRouteDecision:
        ingest_like_decision = self.route_ingest(question)
        return QueryRouteDecision(
            target_path=ingest_like_decision.target_path,
            allowed_context=AllowedContext(
                include_ancestors=True,
                include_peer_links=True,
                exclude_other_branches=True,
            ),
            query_type=self._infer_query_type(question),
            confidence=ingest_like_decision.confidence,
            reasoning_summary=ingest_like_decision.reasoning_summary.replace("Matched", "Query matched"),
        )

    def _infer_query_type(self, question: str) -> QueryType:
        lowered = question.lower()
        if "compare" in lowered or "difference" in lowered or " vs " in lowered:
            return "comparison"
        if "summarize" in lowered or "summary" in lowered:
            return "summary"
        if lowered.startswith(("how ", "why ", "what ")):
            return "explanation"
        if lowered.endswith("?"):
            return "lookup"
        return "unknown"

    def _matches_databricks(self, lowered: str) -> bool:
        return any(
            keyword in lowered
            for keyword in [
                "databricks",
                "delta",
                "spark",
                "structured streaming",
                "auto loader",
                "autoloader",
                "cloudfiles",
                "mergeschema",
            ]
        )

    def _route_databricks(self, text: str) -> RouteDecision:
        lowered = text.lower()
        path = "WORK/DataArt/Databricks"
        peer_links: list[PeerLinkCandidate] = []
        stale_candidates: list[StaleCandidate] = []
        confidence = 0.75

        is_schema_evolution = "schema evolution" in lowered or "mergeschema" in lowered
        is_auto_loader = (
            "auto loader" in lowered
            or "autoloader" in lowered
            or "cloudfiles" in lowered
        )
        is_streaming = "structured streaming" in lowered or "streaming" in lowered
        is_delta_sink = "delta" in lowered or "sink" in lowered or "mergeschema" in lowered

        if is_schema_evolution and is_auto_loader:
            path = "WORK/DataArt/Databricks/Certification/AutoLoader/SchemaEvolution"
            peer_links.append(
                PeerLinkCandidate(
                    path="WORK/DataArt/Databricks/Certification/StructuredStreaming/SchemaEvolution",
                    reason="Auto Loader schema evolution is commonly compared with Delta streaming sink schema evolution.",
                )
            )
            confidence = 0.9
        elif is_schema_evolution and (is_streaming or is_delta_sink):
            path = "WORK/DataArt/Databricks/Certification/StructuredStreaming/SchemaEvolution"
            peer_links.append(
                PeerLinkCandidate(
                    path="WORK/DataArt/Databricks/Certification/AutoLoader/SchemaEvolution",
                    reason="Delta streaming sink schema evolution is commonly confused with Auto Loader schema evolution.",
                )
            )
            stale_candidates.append(
                StaleCandidate(
                    path=path,
                    reason="Older notes may confuse cloudFiles schema evolution settings with Delta sink writes.",
                )
            )
            confidence = 0.9
        elif is_schema_evolution:
            path = "WORK/DataArt/Databricks/Certification/StructuredStreaming/SchemaEvolution"
            peer_links.append(
                PeerLinkCandidate(
                    path="WORK/DataArt/Databricks/Certification/AutoLoader/SchemaEvolution",
                    reason="Generic Databricks schema evolution questions should keep Auto Loader as an approved comparison peer.",
                )
            )
            confidence = 0.8
        elif is_auto_loader:
            path = "WORK/DataArt/Databricks/Certification/AutoLoader"
            confidence = 0.85
        elif is_streaming:
            path = "WORK/DataArt/Databricks/Certification/StructuredStreaming"
            confidence = 0.85
        elif "certification" in lowered or "exam" in lowered:
            path = "WORK/DataArt/Databricks/Certification"
            confidence = 0.8

        return RouteDecision(
            target_path=path,
            content_type=self._infer_content_type(lowered),
            layer="silver",
            action="append_and_optimize",
            peer_links=peer_links,
            stale_candidates=stale_candidates,
            confidence=confidence,
            reasoning_summary=f"Matched Databricks concepts and selected deepest relevant path: {path}.",
        )

    def _infer_content_type(self, lowered: str) -> ContentType:
        if any(keyword in lowered for keyword in ["correction", "instead", "not ", "use ", "should"]):
            return "correction"
        if any(keyword in lowered for keyword in ["decided", "decision"]):
            return "decision"
        if "?" in lowered:
            return "question"
        return "fact"
