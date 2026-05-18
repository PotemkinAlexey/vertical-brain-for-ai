"""Tests for LSH-accelerated namespace similarity search."""
from __future__ import annotations

from vertical_brain.core.models import Chunk
from vertical_brain.core.optimizer import SimpleOptimizer
from vertical_brain.core.vector_lsh import find_similar_pairs
from vertical_brain.llm.embedding import MockEmbeddingProvider
from vertical_brain.storage.json_store import JsonStore

MIN_PARTS = 5
SIMILAR_TEXT = "machine learning model training pipeline gradient descent"


def test_find_similar_pairs_brute_force_finds_identical_vectors():
    vec = [1.0, 0.0, 0.0]
    pairs, stats = find_similar_pairs(
        {"A": vec, "B": list(vec), "C": [0.0, 1.0, 0.0]},
        threshold=0.99,
        brute_force_max=10,
    )
    assert stats.method == "brute_force"
    assert {(a, b) for a, b, _ in pairs} == {("A", "B")}


def test_lsh_finds_similar_pair_among_many_namespaces():
    provider = MockEmbeddingProvider()
    path_vectors: dict[str, list[float]] = {}
    for i in range(400):
        text = f"random namespace topic number {i} alpha beta gamma"
        path_vectors[f"PROJECTS/ns{i:04d}"] = provider.embed(text)
    path_vectors["PROJECTS/anchor_a"] = provider.embed(SIMILAR_TEXT)
    path_vectors["PROJECTS/anchor_b"] = provider.embed(SIMILAR_TEXT)

    pairs, stats = find_similar_pairs(
        path_vectors,
        threshold=0.5,
        brute_force_max=64,
        num_planes=16,
        num_bands=4,
    )
    assert stats.method == "lsh"
    assert stats.namespace_count == 402
    assert stats.candidate_pairs < stats.namespace_count * (stats.namespace_count - 1) // 2
    linked = {(a, b) for a, b, _ in pairs}
    assert ("PROJECTS/anchor_a", "PROJECTS/anchor_b") in linked


def test_discover_links_report_includes_lsh_stats(tmp_path):
    store = JsonStore(tmp_path)
    for i in range(120):
        store.save_chunk(
            Chunk(
                node_path=f"PROJECTS/ns{i:03d}",
                content=f"topic filler text number {i}",
                layer="silver",
            )
        )
    store.save_chunk(Chunk(node_path="PROJECTS/alpha", content=SIMILAR_TEXT, layer="silver"))
    store.save_chunk(Chunk(node_path="PROJECTS/beta", content=SIMILAR_TEXT, layer="silver"))

    optimizer = SimpleOptimizer(
        store,
        min_compaction_path_parts=MIN_PARTS,
        embedding_provider=MockEmbeddingProvider(),
        link_similarity_threshold=0.5,
        link_discovery_brute_force_max=64,
    )
    result = optimizer.discover_links()

    assert "[lsh:" in result
    assert store.list_links()
