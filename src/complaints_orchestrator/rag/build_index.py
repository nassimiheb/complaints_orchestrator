"""Build a local Chroma index from internal policy documents."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from chromadb import PersistentClient

from complaints_orchestrator.rag.local_embeddings import build_embedding_model, resolve_embedding_provider
from complaints_orchestrator.utils.rag_security import (
    chunk_text,
    contains_prompt_injection,
    infer_document_metadata,
    is_allowed_source,
    sanitize_rag_text,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_COLLECTION_NAME = "internal_policy_docs"


def build_index(
    docs_dir: str,
    chroma_dir: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    max_chunk_chars: int = 700,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_api_key: str | None = None,
) -> dict[str, int]:
    docs_root = Path(docs_dir).resolve()
    if not docs_root.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {docs_root}")

    chroma_path = Path(chroma_dir).resolve()
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=collection_name)
    resolved_provider = resolve_embedding_provider(embedding_provider)
    embedder = build_embedding_model(
        provider=resolved_provider,
        model_name=embedding_model,
        explicit_api_key=embedding_api_key,
    )
    LOGGER.info("Embedding provider for indexing: %s", resolved_provider)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    total_docs = 0
    indexed_chunks = 0
    skipped_chunks = 0

    for doc_path in sorted(docs_root.rglob("*")):
        if not is_allowed_source(doc_path, docs_root):
            continue

        total_docs += 1
        raw_text = doc_path.read_text(encoding="utf-8")
        metadata = infer_document_metadata(doc_path)
        relative_source = str(doc_path.relative_to(docs_root))

        raw_chunks = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_index, raw_chunk in enumerate(raw_chunks):
            if contains_prompt_injection(raw_chunk):
                skipped_chunks += 1
                LOGGER.warning("Skipped suspicious chunk during indexing: %s#%s", relative_source, chunk_index)
                continue

            sanitized = sanitize_rag_text(raw_chunk, max_chars=max_chunk_chars)
            if not sanitized:
                skipped_chunks += 1
                continue
            if contains_prompt_injection(sanitized):
                skipped_chunks += 1
                LOGGER.warning("Skipped suspicious sanitized chunk during indexing: %s#%s", relative_source, chunk_index)
                continue

            chunk_id = f"{metadata['doc_id']}::{chunk_index}"
            ids.append(chunk_id)
            documents.append(sanitized)
            metadatas.append(
                {
                    **metadata,
                    "source_path": relative_source,
                    "chunk_index": chunk_index,
                }
            )

    if documents:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embedder.embed_documents(documents),
        )
        indexed_chunks = len(documents)

    return {
        "documents_seen": total_docs,
        "indexed_chunks": indexed_chunks,
        "skipped_chunks": skipped_chunks,
    }


def _default_documents_dir() -> str:
    return str(Path(__file__).resolve().parent / "documents")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build secure Chroma policy index.")
    parser.add_argument("--docs-dir", default=_default_documents_dir(), help="Path to internal policy documents.")
    parser.add_argument("--chroma-dir", default="./storage/chroma", help="Path to local Chroma directory.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="Chunk overlap in characters.")
    parser.add_argument("--max-chunk-chars", type=int, default=700, help="Max sanitized chunk length.")
    parser.add_argument(
        "--embedding-provider",
        default=os.getenv("CCO_EMBEDDING_PROVIDER", "hash"),
        help="Embedding provider: hash or mistral.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("CCO_EMBEDDING_MODEL", "mistral-embed"),
        help="Embedding model name (used by provider).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stats = build_index(
        docs_dir=args.docs_dir,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_chunk_chars=args.max_chunk_chars,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
    )
    print(f"Index build complete: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
