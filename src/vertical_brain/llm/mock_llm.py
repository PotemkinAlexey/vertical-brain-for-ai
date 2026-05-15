from __future__ import annotations

import json
from pathlib import Path


class MockLLM:
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []

    @classmethod
    def from_response_file(cls, path: str | Path) -> MockLLM:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return cls([json.dumps(item, ensure_ascii=False) for item in payload])
        return cls([json.dumps(payload, ensure_ascii=False)])

    def complete(self, prompt: str) -> str:
        if self.responses:
            return self.responses.pop(0)

        payload = json.loads(prompt)
        if payload.get("mode") == "answer":
            context = payload.get("locked_context", [])
            if not context:
                return "I do not have allowed context to answer this question yet."
            lines = ["Based only on locked context:"]
            for item in context[:3]:
                lines.append(f"- {item}")
            return "\n".join(lines)

        if payload.get("mode") == "ask":
            return json.dumps(
                {
                    "target_path": "INBOX/Unclassified",
                    "allowed_context": {
                        "include_ancestors": True,
                        "include_peer_links": True,
                        "exclude_other_branches": True,
                    },
                    "query_type": "unknown",
                    "confidence": 0.0,
                    "reasoning_summary": "No routing model provider configured.",
                }
            )

        return json.dumps(
            {
                "target_path": "INBOX/Unclassified",
                "content_type": "note",
                "layer": "bronze",
                "action": "ask_clarification",
                "peer_links": [],
                "stale_candidates": [],
                "confidence": 0.0,
                "reasoning_summary": "No routing model provider configured.",
            }
        )

    def answer_from_context(self, question: str, context: list[str]) -> str:
        return self.complete(
            json.dumps(
                {
                    "mode": "answer",
                    "question": question,
                    "locked_context": context,
                    "instruction": "Answer only from locked_context.",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
