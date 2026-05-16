from __future__ import annotations

import json


def parse_gold_content(content: str) -> list[str]:
    """Parse Gold chunk content into a list of aspects.

    Supports both formats for backward compatibility:
    - Plain text: ``"Delta migration | AutoLoader streaming"``
    - Structured JSON: ``{"aspects": ["Delta migration", ...], "last_updated": "..."}``
    """
    text = content.strip()
    if not text:
        return []
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("aspects"), list):
            return [str(aspect) for aspect in parsed["aspects"] if str(aspect).strip()]
    return [aspect.strip() for aspect in text.split(" | ") if aspect.strip()]
