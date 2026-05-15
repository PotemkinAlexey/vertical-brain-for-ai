from __future__ import annotations

from vertical_brain.core.models import RouteDecision, PeerLinkCandidate


class MockRouter:
    """Simple deterministic router for MVP v0.1.

    This should later be replaced by an LLM router that returns the same
    RouteDecision contract.
    """

    def route_ingest(self, text: str) -> RouteDecision:
        lowered = text.lower()

        if "databricks" in lowered or "delta" in lowered or "spark" in lowered:
            path = "WORK/DataArt/Databricks"
            peers = []
            if "auto loader" in lowered or "autoloader" in lowered or "cloudfiles" in lowered:
                peers.append(
                    PeerLinkCandidate(
                        path="WORK/DataArt/Databricks/AutoLoader",
                        reason="Input mentions Auto Loader or cloudFiles concepts.",
                    )
                )
            return RouteDecision(
                target_path=path,
                content_type="fact",
                layer="silver",
                action="append_and_optimize",
                peer_links=peers,
                confidence=0.75,
                reasoning_summary="Matched Databricks/Spark/Delta keywords.",
            )

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
            action="append_bronze",
            confidence=0.4,
            reasoning_summary="No strong route found.",
        )

    def route_query(self, question: str) -> RouteDecision:
        return self.route_ingest(question)
