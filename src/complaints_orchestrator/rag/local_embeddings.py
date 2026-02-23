"""Embedding providers for Chroma indexing/retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any, Callable, Protocol
from urllib import error, request

from complaints_orchestrator.utils.mistral import resolve_mistral_api_key

MISTRAL_EMBEDDINGS_URL = "https://api.mistral.ai/v1/embeddings"
DEFAULT_EMBEDDING_PROVIDER = "hash"
DEFAULT_MISTRAL_EMBEDDING_MODEL = "mistral-embed"


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class HashEmbeddingModel:
    """Deterministic local embeddings for offline usage and tests."""

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


class MistralEmbeddingModel:
    """Mistral-backed embedding model for production semantic retrieval."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MISTRAL_EMBEDDING_MODEL,
        timeout_seconds: int = 30,
        batch_size: int = 32,
        urlopen_fn: Callable[..., Any] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size
        self.urlopen_fn = urlopen_fn or request.urlopen

    def _request_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": texts,
        }
        req = request.Request(
            url=MISTRAL_EMBEDDINGS_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with self.urlopen_fn(req, timeout=self.timeout_seconds) as resp:
                raw_response = resp.read().decode("utf-8")
        except (error.URLError, error.HTTPError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Mistral embeddings call failed: {exc}") from exc

        try:
            parsed = json.loads(raw_response)
            rows = parsed["data"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Invalid Mistral embeddings response format: {exc}") from exc

        if not isinstance(rows, list):
            raise RuntimeError("Invalid Mistral embeddings response format: data must be a list.")

        embeddings: list[list[float]] = []
        for row in sorted(rows, key=lambda item: int(item.get("index", 0))):
            raw_embedding = row.get("embedding")
            if not isinstance(raw_embedding, list):
                raise RuntimeError("Invalid Mistral embeddings response format: embedding must be a list.")
            try:
                vector = [float(value) for value in raw_embedding]
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Invalid Mistral embeddings response format: non-numeric values.") from exc
            embeddings.append(vector)

        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Invalid Mistral embeddings response format: expected {len(texts)} vectors, got {len(embeddings)}."
            )
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        output: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            output.extend(self._request_batch(batch))
        return output

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embed_documents([text])
        if not embeddings:
            return []
        return embeddings[0]


def resolve_embedding_provider(explicit_provider: str | None = None) -> str:
    provider = explicit_provider or os.getenv("CCO_EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER)
    return provider.strip().lower()


def resolve_embedding_model_name(explicit_model: str | None = None) -> str:
    if explicit_model and explicit_model.strip():
        return explicit_model.strip()
    return os.getenv("CCO_EMBEDDING_MODEL", DEFAULT_MISTRAL_EMBEDDING_MODEL)


def build_embedding_model(
    *,
    provider: str | None = None,
    model_name: str | None = None,
    explicit_api_key: str | None = None,
    timeout_seconds: int = 30,
    batch_size: int = 32,
    urlopen_fn: Callable[..., Any] | None = None,
) -> Embedder:
    resolved_provider = resolve_embedding_provider(provider)
    if resolved_provider in {"hash", "local", "deterministic"}:
        return HashEmbeddingModel()

    if resolved_provider == "mistral":
        api_key = resolve_mistral_api_key(
            explicit_api_key,
            "MISTRAL_API_KEY is required for Mistral embeddings.",
        )
        return MistralEmbeddingModel(
            api_key=api_key,
            model=resolve_embedding_model_name(model_name),
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            urlopen_fn=urlopen_fn,
        )

    raise ValueError(f"Unsupported embedding provider: {resolved_provider}")
