"""Configuration loading for the orchestrator."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return False


@dataclass(frozen=True)
class AppConfig:
    llm_provider: str
    mistral_api_key: str
    model_name: str
    embedding_model: str
    chroma_dir: str
    sqlite_path: str
    hitl_amount_threshold: float
    low_confidence_threshold: float
    log_level: str

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "AppConfig":
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)

        return cls(
            llm_provider=os.getenv("CCO_LLM_PROVIDER", "mistral").lower(),
            mistral_api_key=os.getenv("MISTRAL_API_KEY", ""),
            model_name=os.getenv("CCO_MODEL_NAME", "mistral-small-latest"),
            embedding_model=os.getenv("CCO_EMBEDDING_MODEL", "mistral-embed"),
            chroma_dir=os.getenv("CCO_CHROMA_DIR", "./storage/chroma"),
            sqlite_path=os.getenv("CCO_SQLITE_PATH", "./storage/complaints_memory.db"),
            hitl_amount_threshold=float(os.getenv("CCO_HITL_AMOUNT_THRESHOLD", "150.0")),
            low_confidence_threshold=float(os.getenv("CCO_LOW_CONFIDENCE_THRESHOLD", "0.55")),
            log_level=os.getenv("CCO_LOG_LEVEL", "INFO").upper(),
        )
