"""Shared local JSON data loading for mock tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _data_dir() -> Path:
    return _project_root() / "data"


def load_json_records(file_name: str) -> list[dict[str, Any]]:
    path = _data_dir() / file_name
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of records in {path}")
    return data

