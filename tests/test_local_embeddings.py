"""Tests for embedding provider resolution and Mistral embedding adapter."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from complaints_orchestrator.rag.local_embeddings import (  # noqa: E402
    HashEmbeddingModel,
    MistralEmbeddingModel,
    build_embedding_model,
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestLocalEmbeddings(unittest.TestCase):
    def test_build_embedding_model_defaults_to_hash(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            embedder = build_embedding_model()
        self.assertIsInstance(embedder, HashEmbeddingModel)

    def test_build_embedding_model_mistral_requires_key(self) -> None:
        with patch.dict(os.environ, {"CCO_EMBEDDING_PROVIDER": "mistral"}, clear=True):
            with self.assertRaises(RuntimeError):
                build_embedding_model(provider="mistral")

    def test_mistral_embedding_model_embed_documents_batches(self) -> None:
        calls: list[list[str]] = []

        def _mock_urlopen(req, timeout=30):
            body = json.loads(req.data.decode("utf-8"))
            inputs = body["input"]
            calls.append(inputs)
            payload = {
                "data": [
                    {"index": idx, "embedding": [float(idx), float(len(text))]}
                    for idx, text in enumerate(inputs)
                ]
            }
            return _FakeHTTPResponse(payload)

        model = MistralEmbeddingModel(
            api_key="test-key",
            model="mistral-embed",
            batch_size=2,
            urlopen_fn=_mock_urlopen,
        )
        vectors = model.embed_documents(["alpha", "beta", "gamma"])

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], ["alpha", "beta"])
        self.assertEqual(calls[1], ["gamma"])
        self.assertEqual(vectors, [[0.0, 5.0], [1.0, 4.0], [0.0, 5.0]])


if __name__ == "__main__":
    unittest.main()
