from vertical_brain.core.models import Chunk, SearchResult
from vertical_brain.core.search import rank_paths_from_results
from vertical_brain.core.context_session import ContextSession
from vertical_brain.storage.json_store import JsonStore


def _result(path, score):
    return SearchResult(path=path, source="chunk", score=score, snippet="")


def test_rank_paths_picks_path_with_most_hits_over_single_hit():
    # Both paths have a comparable top score, but B has more supporting hits,
    # so the hit-count component of the composite score breaks the tie for B.
    results = [
        _result("WORK/A", 0.85),
        _result("WORK/B", 0.85),
        _result("WORK/B", 0.85),
        _result("WORK/B", 0.85),
    ]
    ranks = rank_paths_from_results(results)
    assert ranks[0].path == "WORK/B"
    assert ranks[1].path == "WORK/A"
    assert ranks[0].hit_count == 3


def test_rank_paths_is_deterministic():
    results = [
        _result("WORK/A", 0.5),
        _result("WORK/B", 0.5),
        _result("WORK/A", 0.5),
    ]
    first = rank_paths_from_results(results)
    second = rank_paths_from_results(results)
    assert [r.path for r in first] == [r.path for r in second]
    assert [r.score for r in first] == [r.score for r in second]


def test_context_search_uses_path_ranking(tmp_path):
    store = JsonStore(tmp_path)
    for _ in range(3):
        store.save_chunk(Chunk(node_path="WORK/Multi", content="alpha keyword fact"))
    store.save_chunk(Chunk(node_path="WORK/Single", content="alpha keyword fact"))

    result = ContextSession(store).search_locked_context(
        "alpha keyword",
        search_limit=20,
        context_limit=2,
    )
    paths = [c.target_path for c in result.locked_contexts]
    assert paths[0] == "WORK/Multi"
