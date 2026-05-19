"""Embedding providers and the cosine similarity helper.

`EmbeddingProvider` is the contract that `EmbeddingSearch` /
`EmbeddingRouter` rely on. The contract is intentionally narrow:

- `model_name: str` — used to key the persistent vector cache so a model
  switch does not silently reuse stale vectors of a different shape.
- `embed_dimension: int` — `-1` when the dimension is only knowable
  after the first `embed()` call.
- `is_semantic: bool` — `False` for bag-of-words baselines like Mock.
  The read-path uses this flag to decide whether to apply semantic
  upgrades (`silver_confidence` weak → ok via cosine) and how to phrase
  agent-facing hints.
- `embed(text) -> list[float]` — single-text embedding.
- `embed_batch(texts) -> list[list[float]]` — batch embedding. Native
  implementations should issue a single API call; the default in
  `BaseEmbeddingProvider` falls back to N×`embed` so single-text
  providers keep working.

`BaseEmbeddingProvider` provides safe defaults for new providers:
`is_semantic = True`, and an `embed_batch` that delegates to `embed`.
Subclasses must implement `embed` and should set `model_name` /
`embed_dimension` (as plain attributes or properties).

Consumers in this package (`EmbeddingSearch`, `EmbeddingRouter`) read
provider attributes via `getattr` with sensible fallbacks, so pre-v1.9
providers that do not inherit `BaseEmbeddingProvider` continue to work.
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from typing import Protocol, runtime_checkable


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


@runtime_checkable
class EmbeddingProvider(Protocol):
    model_name: str
    embed_dimension: int
    is_semantic: bool

    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class BaseEmbeddingProvider:
    """Default implementation surface for `EmbeddingProvider` implementers.

    Subclasses must override `embed`. They may override `embed_batch` to
    use a native batch endpoint (OpenAI / Cohere / Voyage all accept
    `input: list[str]` in one request); the default below issues N×embed
    calls so single-text providers work out of the box.

    Subclasses should set `model_name` (used as the vector cache key) and
    `embed_dimension` (when known) on the instance. `is_semantic`
    defaults to `True`; bag-of-words baselines like Mock override to
    `False`.
    """

    is_semantic: bool = True

    def embed(self, text: str) -> list[float]:  # pragma: no cover - abstract
        raise NotImplementedError("EmbeddingProvider subclasses must implement embed()")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Bag-of-words pseudo-embeddings for tests — similar text → higher similarity."""

    DIM = 64
    is_semantic = False

    def __init__(self, model_name: str = "mock-v1") -> None:
        self.model_name = model_name

    @property
    def embed_dimension(self) -> int:
        return self.DIM

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM
            vec[idx] += 1.0
        mag = math.sqrt(sum(v * v for v in vec))
        return [v / mag for v in vec] if mag > 0 else vec


class HttpEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI-compatible embeddings endpoint (works with OpenAI, Ollama, etc.).

    Overrides `embed_batch` to send the whole list to the endpoint in one
    request — OpenAI / Ollama / Cohere accept `input: list[str]` and
    return one vector per element. This collapses N round-trips into one.
    """

    is_semantic = True

    def __init__(self, url: str, model: str, api_key: str = "", timeout: float = 30.0):
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def embed_dimension(self) -> int:
        return -1  # not known until first embed call

    def embed(self, text: str) -> list[float]:
        result = self._post({"input": text, "model": self.model})
        try:
            embedding = result["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Embedding endpoint returned an invalid OpenAI-compatible response") from exc
        return self._coerce_vector(embedding)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self._post({"input": list(texts), "model": self.model})
        entries = result.get("data") if isinstance(result, dict) else None
        if not isinstance(entries, list) or len(entries) != len(texts):
            raise ValueError(
                "Embedding endpoint returned an invalid OpenAI-compatible batch response "
                f"(expected {len(texts)} vectors, got "
                f"{len(entries) if isinstance(entries, list) else 'non-list'})"
            )
        return [self._coerce_vector(entry.get("embedding") if isinstance(entry, dict) else None)
                for entry in entries]

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    @staticmethod
    def _coerce_vector(embedding: object) -> list[float]:
        if not isinstance(embedding, list) or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in embedding
        ):
            raise ValueError("Embedding endpoint returned a non-numeric embedding vector")
        return [float(v) for v in embedding]
