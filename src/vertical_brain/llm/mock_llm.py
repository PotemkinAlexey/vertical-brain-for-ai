from __future__ import annotations

import re


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "why",
    "with",
    "work",
}


class MockLLM:
    def complete(self, prompt: str) -> str:
        return "Mock response. Replace with real LLM provider later."

    def answer_from_context(self, question: str, context: list[str]) -> str:
        if not context:
            return "I do not have allowed context to answer this question yet."

        question_tokens = self._tokens(question)
        ranked_context = sorted(
            (
                (len(question_tokens & self._tokens(item)), index, item)
                for index, item in enumerate(context)
            ),
            key=lambda scored_item: (-scored_item[0], scored_item[1]),
        )
        selected = [
            item
            for score, _index, item in ranked_context
            if not question_tokens or score > 0
        ]
        if not selected:
            selected = [item for _score, _index, item in ranked_context]

        lines = ["Based only on locked context:"]
        for item in selected[:3]:
            lines.append(f"- {item}")
        return "\n".join(lines)

    def _tokens(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9_]+", text.lower())
            if len(token) > 1 and token not in STOP_WORDS
        }
