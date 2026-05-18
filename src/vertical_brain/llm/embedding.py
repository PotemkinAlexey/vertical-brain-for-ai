from __future__ import annotations

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
    def embed(self, text: str) -> list[float]: ...


class MockEmbeddingProvider:
    """Bag-of-words pseudo-embeddings for tests — similar text → higher similarity."""

    DIM = 64

    def __init__(self, model_name: str = "mock-v1") -> None:
        self.model_name = model_name

    @property
    def embed_dimension(self) -> int:
        return self.DIM

    def embed(self, text: str) -> list[float]:
        import hashlib
        vec = [0.0] * self.DIM
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM
            vec[idx] += 1.0
        mag = math.sqrt(sum(v * v for v in vec))
        return [v / mag for v in vec] if mag > 0 else vec


class HttpEmbeddingProvider:
    """OpenAI-compatible embeddings endpoint (works with OpenAI, Ollama, etc.)."""

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
        payload = json.dumps({"input": text, "model": self.model}).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())
        try:
            embedding = result["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Embedding endpoint returned an invalid OpenAI-compatible response") from exc
        if not isinstance(embedding, list) or not all(isinstance(v, (int, float)) for v in embedding):
            raise ValueError("Embedding endpoint returned a non-numeric embedding vector")
        return [float(v) for v in embedding]
