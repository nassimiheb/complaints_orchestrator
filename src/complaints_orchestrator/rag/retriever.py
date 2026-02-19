"""Secure Chroma retriever for policy snippets."""

from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import Any

from chromadb import PersistentClient

from complaints_orchestrator.rag.build_index import DEFAULT_COLLECTION_NAME
from complaints_orchestrator.rag.local_embeddings import HashEmbeddingModel
from complaints_orchestrator.utils.rag_security import contains_prompt_injection, sanitize_rag_text

LOGGER = logging.getLogger(__name__)


def _is_internal_source(source_path: str) -> bool:
    path = PurePosixPath(source_path.replace("\\", "/"))
    if path.is_absolute():
        return False
    if ".." in path.parts:
        return False
    return True


class PolicyRetriever:
    def __init__(
        self,
        chroma_dir: str,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        max_excerpt_chars: int = 400,
    ) -> None:
        self.client = PersistentClient(path=chroma_dir)
        self.collection = self.client.get_collection(name=collection_name)
        self.embedder = HashEmbeddingModel()
        self.max_excerpt_chars = max_excerpt_chars

    def retrieve(
        self,
        query: str,
        language: str,
        top_k: int = 4,
        policy_type: str | None = None,
    ) -> list[dict[str, Any]]:
        sanitized_query = sanitize_rag_text(query, max_chars=300)
        if not sanitized_query:
            return []

        query_embedding = self.embedder.embed_query(sanitized_query)
        desired_language = language.upper()
        desired_policy_type = policy_type.upper() if policy_type else None
        fetch_k = max(top_k * 3, top_k)

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            where={"language": desired_language},
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        output: list[dict[str, Any]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            source_path = str(metadata.get("source_path", ""))
            if not _is_internal_source(source_path):
                LOGGER.warning("Skipped non-internal source in retrieval: %s", source_path)
                continue

            if desired_policy_type and str(metadata.get("policy_type", "")).upper() != desired_policy_type:
                continue

            snippet = sanitize_rag_text(str(document), max_chars=self.max_excerpt_chars)
            if not snippet:
                continue
            if contains_prompt_injection(snippet):
                LOGGER.warning("Skipped suspicious chunk during retrieval: %s", source_path)
                continue

            score = 1.0 - float(distance) if distance is not None else 0.0
            output.append(
                {
                    "doc_id": str(metadata.get("doc_id", "")),
                    "language": str(metadata.get("language", "")),
                    "policy_type": str(metadata.get("policy_type", "")),
                    "source_path": source_path,
                    "snippet": snippet,
                    "score": score,
                }
            )
            if len(output) >= top_k:
                break

        return output

