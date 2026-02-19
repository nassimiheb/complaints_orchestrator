"""Deterministic local embeddings for offline Chroma indexing/retrieval."""

from __future__ import annotations

import hashlib
import math
import re


class HashEmbeddingModel:
    def __init__(self, dimensions: int = 96) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be > 0")
        self.dimensions = dimensions

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = self._tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, 12):
                index = digest[i] % self.dimensions
                sign = 1.0 if digest[i + 12] % 2 == 0 else -1.0
                vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

