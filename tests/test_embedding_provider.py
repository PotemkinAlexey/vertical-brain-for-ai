import json

import pytest

from vertical_brain.llm.embedding import (
    BaseEmbeddingProvider,
    HttpEmbeddingProvider,
    MockEmbeddingProvider,
)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_http_embedding_provider_uses_timeout_and_normalizes_vector(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append((req, timeout))
        return _Response({"data": [{"embedding": [1, 2.5, 3]}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    provider = HttpEmbeddingProvider("http://embedding.test/v1/embeddings", "test-model", timeout=4.5)

    assert provider.embed("hello") == [1.0, 2.5, 3.0]
    assert calls[0][1] == 4.5


def test_http_embedding_provider_rejects_invalid_response(monkeypatch):
    def fake_urlopen(req, timeout):
        return _Response({"unexpected": []})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    provider = HttpEmbeddingProvider("http://embedding.test/v1/embeddings", "test-model")

    with pytest.raises(ValueError, match="invalid OpenAI-compatible response"):
        provider.embed("hello")


# ---------------------------------------------------------------------------
# v1.9: BaseEmbeddingProvider defaults
# ---------------------------------------------------------------------------


def test_base_embedding_provider_is_semantic_defaults_true():
    class Custom(BaseEmbeddingProvider):
        model_name = "custom"
        embed_dimension = 3
        def embed(self, text):
            return [1.0, 0.0, 0.0]

    assert Custom().is_semantic is True


def test_base_embedding_provider_embed_must_be_overridden():
    base = BaseEmbeddingProvider()
    with pytest.raises(NotImplementedError):
        base.embed("hello")


def test_base_embedding_provider_default_embed_batch_delegates_to_embed():
    calls: list[str] = []

    class Counting(BaseEmbeddingProvider):
        model_name = "counting"
        embed_dimension = 1
        def embed(self, text):
            calls.append(text)
            return [float(len(text))]

    provider = Counting()
    vecs = provider.embed_batch(["a", "bb", "ccc"])
    assert vecs == [[1.0], [2.0], [3.0]]
    assert calls == ["a", "bb", "ccc"]


def test_mock_embedding_provider_is_not_semantic():
    assert MockEmbeddingProvider().is_semantic is False


def test_mock_embedding_provider_inherits_default_embed_batch():
    provider = MockEmbeddingProvider()
    batch = provider.embed_batch(["alpha", "beta"])
    assert batch == [provider.embed("alpha"), provider.embed("beta")]


# ---------------------------------------------------------------------------
# v1.9: HttpEmbeddingProvider native batch endpoint
# ---------------------------------------------------------------------------


def test_http_embedding_provider_is_semantic_true():
    assert HttpEmbeddingProvider("http://x/v1/embeddings", "m").is_semantic is True


def test_http_embedding_provider_embed_batch_single_request(monkeypatch):
    captured: dict = {}

    def fake_urlopen(req, timeout):
        captured["data"] = json.loads(req.data.decode())
        return _Response({"data": [
            {"embedding": [1.0, 0.0]},
            {"embedding": [0.0, 1.0]},
            {"embedding": [0.5, 0.5]},
        ]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = HttpEmbeddingProvider("http://embedding.test/v1/embeddings", "test-model")

    vecs = provider.embed_batch(["alpha", "beta", "gamma"])

    assert vecs == [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    assert captured["data"]["input"] == ["alpha", "beta", "gamma"]
    assert captured["data"]["model"] == "test-model"


def test_http_embedding_provider_embed_batch_empty_input_no_request(monkeypatch):
    def fake_urlopen(req, timeout):
        raise AssertionError("urlopen should not be called for empty batch")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = HttpEmbeddingProvider("http://embedding.test/v1/embeddings", "m")

    assert provider.embed_batch([]) == []


def test_http_embedding_provider_embed_batch_rejects_count_mismatch(monkeypatch):
    def fake_urlopen(req, timeout):
        return _Response({"data": [{"embedding": [1.0]}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = HttpEmbeddingProvider("http://embedding.test/v1/embeddings", "m")

    with pytest.raises(ValueError, match="invalid OpenAI-compatible batch response"):
        provider.embed_batch(["a", "b"])
