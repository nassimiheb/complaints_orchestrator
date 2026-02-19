"""Security helpers for RAG indexing and retrieval."""

from __future__ import annotations

import re
from pathlib import Path

ALLOWED_DOCUMENT_EXTENSIONS = {".md", ".txt"}
SUSPICIOUS_PATTERNS = (
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"reveal\s+.*system\s+prompt",
    r"developer\s+message",
    r"system\s+prompt",
    r"tool\s*call",
    r"execute\s+shell",
    r"<script",
    r"BEGIN\s+INJECTION",
)
DIRECTIVE_LINE_PATTERNS = (
    r"^\s*(system|assistant|developer)\s*:\s*",
    r"^\s*(ignore|override)\b",
    r"^\s*(execute|run)\s+",
)


def is_allowed_source(path: Path, allow_root: Path) -> bool:
    resolved_path = path.resolve()
    resolved_root = allow_root.resolve()
    if resolved_path.suffix.lower() not in ALLOWED_DOCUMENT_EXTENSIONS:
        return False
    if not resolved_path.is_file():
        return False
    return resolved_root == resolved_path or resolved_root in resolved_path.parents


def contains_prompt_injection(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in SUSPICIOUS_PATTERNS)


def strip_directive_like_lines(text: str) -> str:
    kept_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")
            continue
        if any(re.search(pattern, stripped.lower()) for pattern in DIRECTIVE_LINE_PATTERNS):
            continue
        if contains_prompt_injection(stripped):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def sanitize_rag_text(text: str, max_chars: int) -> str:
    cleaned = strip_directive_like_lines(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0]
    return cleaned


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = end - chunk_overlap
    return chunks


def infer_document_metadata(doc_path: Path) -> dict[str, str]:
    stem = doc_path.stem.lower()
    language = "FR" if stem.endswith("_fr") else "EN"
    policy_type = stem.rsplit("_", 1)[0].upper()
    doc_id = stem.upper().replace("-", "_")
    return {
        "doc_id": doc_id,
        "language": language,
        "policy_type": policy_type,
    }

