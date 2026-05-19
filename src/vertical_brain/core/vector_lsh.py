"""Approximate cosine similarity for many namespace vectors (stdlib only).

Uses random-hyperplane LSH with banding to cut candidate pairs before exact
cosine verification. Falls back to brute force when the namespace count is small.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from vertical_brain.llm.embedding import cosine_similarity


def normalize_vector(vector: list[float]) -> list[float]:
    mag = math.sqrt(sum(x * x for x in vector))
    if mag == 0:
        return list(vector)
    return [x / mag for x in vector]


@dataclass(frozen=True)
class SimilaritySearchStats:
    namespace_count: int
    method: str
    candidate_pairs: int
    matched_pairs: int


def find_similar_pairs(
    path_vectors: dict[str, list[float]],
    *,
    threshold: float,
    brute_force_max: int = 256,
    num_planes: int = 16,
    num_bands: int = 4,
    seed: int = 42,
) -> tuple[list[tuple[str, str, float]], SimilaritySearchStats]:
    """Return (path_a, path_b, score) pairs with path_a < path_b and score >= threshold."""
    items = sorted(
        (path, normalize_vector(vec)) for path, vec in path_vectors.items() if vec
    )
    count = len(items)
    if count < 2:
        return [], SimilaritySearchStats(count, "none", 0, 0)

    if count <= brute_force_max:
        pairs = _brute_force_pairs(items, threshold)
        return pairs, SimilaritySearchStats(count, "brute_force", _pair_count(count), len(pairs))

    candidates = _lsh_candidate_pairs(items, num_planes=num_planes, num_bands=num_bands, seed=seed)
    pairs = _score_candidates(items, candidates, threshold)
    return pairs, SimilaritySearchStats(count, "lsh", len(candidates), len(pairs))


def cluster_by_similarity(
    vectors: dict[str, list[float]],
    *,
    threshold: float,
) -> list[list[str]]:
    """Group keys into clusters by cosine similarity.

    Two keys join the same cluster when their vectors are at least *threshold*
    similar; clusters are the connected components of that graph. Every input
    key appears in exactly one returned list (singletons included). Components
    are sorted largest-first, then by first key.
    """
    keys = sorted(key for key, vec in vectors.items() if vec)
    parent = {key: key for key in keys}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    pairs, _stats = find_similar_pairs(vectors, threshold=threshold)
    for key_a, key_b, _score in pairs:
        root_a, root_b = find(key_a), find(key_b)
        if root_a != root_b:
            parent[root_a] = root_b

    groups: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        groups[find(key)].append(key)
    return sorted(groups.values(), key=lambda group: (-len(group), group[0]))


def _pair_count(n: int) -> int:
    return n * (n - 1) // 2


def _brute_force_pairs(
    items: list[tuple[str, list[float]]],
    threshold: float,
) -> list[tuple[str, str, float]]:
    results: list[tuple[str, str, float]] = []
    for i, (path_a, vec_a) in enumerate(items):
        for path_b, vec_b in items[i + 1 :]:
            score = cosine_similarity(vec_a, vec_b)
            if score >= threshold:
                results.append((path_a, path_b, score))
    return results


def _lsh_candidate_pairs(
    items: list[tuple[str, list[float]]],
    *,
    num_planes: int,
    num_bands: int,
    seed: int,
) -> set[tuple[str, str]]:
    if num_planes < num_bands or num_bands < 1:
        raise ValueError("num_planes must be >= num_bands >= 1")
    dim = len(items[0][1])
    planes = _random_unit_planes(dim, num_planes, seed)
    planes_per_band = num_planes // num_bands

    bands: list[dict[int, list[str]]] = [defaultdict(list) for _ in range(num_bands)]
    for path, vec in items:
        signature = _hyperplane_signature(vec, planes)
        for band_index in range(num_bands):
            shift = band_index * planes_per_band
            mask = (1 << planes_per_band) - 1
            key = (signature >> shift) & mask
            bands[band_index][key].append(path)

    candidates: set[tuple[str, str]] = set()
    for band in bands:
        for bucket in band.values():
            if len(bucket) < 2:
                continue
            ordered = sorted(bucket)
            for i, path_a in enumerate(ordered):
                for path_b in ordered[i + 1 :]:
                    candidates.add((path_a, path_b))
    return candidates


def _score_candidates(
    items: list[tuple[str, list[float]]],
    candidates: Iterable[tuple[str, str]],
    threshold: float,
) -> list[tuple[str, str, float]]:
    by_path = {path: vec for path, vec in items}
    results: list[tuple[str, str, float]] = []
    for path_a, path_b in candidates:
        vec_a = by_path.get(path_a)
        vec_b = by_path.get(path_b)
        if vec_a is None or vec_b is None:
            continue
        score = cosine_similarity(vec_a, vec_b)
        if score >= threshold:
            results.append((path_a, path_b, score))
    return results


def _random_unit_planes(dim: int, count: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    planes: list[list[float]] = []
    for _ in range(count):
        raw = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        planes.append(normalize_vector(raw))
    return planes


def _hyperplane_signature(vector: list[float], planes: list[list[float]]) -> int:
    signature = 0
    for index, plane in enumerate(planes):
        dot = sum(x * y for x, y in zip(vector, plane))
        if dot >= 0:
            signature |= 1 << index
    return signature
