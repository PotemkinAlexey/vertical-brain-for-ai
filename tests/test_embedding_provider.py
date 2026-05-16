import json

import pytest

from vertical_brain.llm.embedding import HttpEmbeddingProvider


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
