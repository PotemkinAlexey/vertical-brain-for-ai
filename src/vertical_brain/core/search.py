from __future__ import annotations

import re

from vertical_brain.core.models import Chunk, Node, SearchResult


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class BrainSearch:
    def __init__(self, store):
        self.store = store

    def search(
        self,
        query: str,
        *,
        root_path: str | None = None,
        limit: int = 10,
        include_stale: bool = False,
    ) -> list[SearchResult]:
        search = getattr(self.store, "search", None)
        if callable(search):
            return search(query, root_path=root_path, limit=limit, include_stale=include_stale)
        return lexical_search(
            nodes=self.store.list_nodes(),
            chunks=self.store.list_chunks(),
            query=query,
            root_path=root_path,
            limit=limit,
            include_stale=include_stale,
        )


def tokenize_query(query: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(query) if token.strip()]


def fts_query(query: str) -> str:
    return " OR ".join(f'"{token}"' for token in tokenize_query(query))


def lexical_search(
    *,
    nodes: list[Node],
    chunks: list[Chunk],
    query: str,
    root_path: str | None = None,
    limit: int = 10,
    include_stale: bool = False,
) -> list[SearchResult]:
    terms = tokenize_query(query)
    if not terms or limit <= 0:
        return []

    results: list[SearchResult] = []
    normalized_query = query.strip().lower()

    for chunk in chunks:
        if not include_stale and chunk.status != "active":
            continue
        if not _path_in_scope(chunk.node_path, root_path):
            continue

        score = _score_text(
            query=normalized_query,
            terms=terms,
            path=chunk.node_path,
            text=chunk.content,
        )
        if score <= 0:
            continue
        results.append(
            SearchResult(
                path=chunk.node_path,
                source="chunk",
                score=float(score),
                snippet=make_snippet(chunk.content, terms),
                chunk_id=chunk.id,
                layer=chunk.layer,
                content_type=chunk.content_type,
                status=chunk.status,
            )
        )

    for node in nodes:
        if not node.gold_summary.strip():
            continue
        if not _path_in_scope(node.path, root_path):
            continue

        score = _score_text(
            query=normalized_query,
            terms=terms,
            path=node.path,
            text=node.gold_summary,
        )
        if score <= 0:
            continue
        results.append(
            SearchResult(
                path=node.path,
                source="gold",
                score=float(score),
                snippet=make_snippet(node.gold_summary, terms),
                layer="gold",
                content_type="summary",
                status="active",
            )
        )

    return sorted(results, key=lambda result: (-result.score, result.path, result.source))[:limit]


def make_snippet(text: str, terms: list[str], *, width: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= width:
        return compact

    lower = compact.lower()
    starts = [lower.find(term) for term in terms if lower.find(term) >= 0]
    if not starts:
        return compact[: width - 3].rstrip() + "..."

    start = max(0, min(starts) - width // 3)
    end = min(len(compact), start + width)
    snippet = compact[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(compact):
        snippet = snippet.rstrip() + "..."
    return snippet


def _score_text(*, query: str, terms: list[str], path: str, text: str) -> int:
    path_lower = path.lower()
    text_lower = text.lower()
    score = 0
    if query:
        if query in text_lower:
            score += 6
        if query in path_lower:
            score += 4

    for term in terms:
        score += text_lower.count(term) * 2
        score += path_lower.count(term) * 3
    return score


def _path_in_scope(path: str, root_path: str | None) -> bool:
    if root_path is None or not root_path:
        return True
    return path == root_path or path.startswith(root_path + "/")
