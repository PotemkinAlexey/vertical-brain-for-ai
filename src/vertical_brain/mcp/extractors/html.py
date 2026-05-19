"""HTML text extraction via stdlib html.parser."""
from __future__ import annotations

import html.parser


class _HTMLTextExtractor(html.parser.HTMLParser):
    """Strip HTML tags and return visible text."""

    _SKIP_TAGS = {"script", "style", "noscript", "head"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self._parts)


def strip_html(raw: str) -> str:
    extractor = _HTMLTextExtractor()
    extractor.feed(raw)
    return extractor.get_text()


def extract_html(source_path: str) -> str:
    with open(source_path, "rb") as f:
        raw = f.read()
    # try utf-8, fall back to latin-1 which never fails
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    content = strip_html(text)
    if not content.strip():
        raise ValueError(f"No text extracted from HTML file: {source_path!r}")
    return content
