"""Service layer for web routes that invoke the orchestration graph."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from complaints_orchestrator.config import AppConfig
from complaints_orchestrator.graph import GraphDependencies, build_dependencies_from_config, run_graph
from complaints_orchestrator.logging_config import configure_logging
from complaints_orchestrator.rag.build_index import DEFAULT_COLLECTION_NAME, build_index
from complaints_orchestrator.state import CaseState
from complaints_orchestrator.web.schemas import RunCaseRequest, RunCaseResponse, ScenarioPreview

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebRuntime:
    """Runtime objects reused across requests."""

    config: AppConfig
    deps: GraphDependencies


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_from_project(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (_project_root() / path).resolve()


def _default_docs_dir() -> Path:
    return _project_root() / "src" / "complaints_orchestrator" / "rag" / "documents"


def _default_scenarios_file() -> Path:
    return _project_root() / "data" / "triage_playground_cases.json"


def _default_eval_scenarios_file() -> Path:
    return _project_root() / "eval" / "scenarios.json"


def _chroma_has_entries(chroma_dir: str) -> bool:
    chroma_path = _resolve_from_project(chroma_dir)
    if not chroma_path.exists():
        return False
    return any(chroma_path.rglob("*"))


def ensure_rag_index_if_missing(config: AppConfig) -> bool:
    """Build the local RAG index only when the configured storage is absent or empty."""

    if _chroma_has_entries(config.chroma_dir):
        return False

    stats = build_index(
        docs_dir=str(_default_docs_dir()),
        chroma_dir=config.chroma_dir,
        collection_name=DEFAULT_COLLECTION_NAME,
    )
    LOGGER.info("RAG index built for web runtime: %s", stats)
    return True


def initialize_runtime(
    env_file: str = ".env",
    ensure_index_if_missing: bool = True,
) -> WebRuntime:
    """Initialize app config and graph dependencies for the web server."""

    config = AppConfig.from_env(env_file=env_file)
    configure_logging(config.log_level)

    if ensure_index_if_missing:
        ensure_rag_index_if_missing(config)

    deps = build_dependencies_from_config(config)
    return WebRuntime(config=config, deps=deps)


def _generate_case_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return f"WEB_CASE_{timestamp}"


def _normalize_case_id(case_id: str | None) -> str:
    raw = case_id or _generate_case_id()
    return raw.strip().upper().replace("-", "_")


def _build_case_state(payload: RunCaseRequest) -> CaseState:
    normalized_case_id = _normalize_case_id(payload.case_id)
    raw_state = {
        "input": {
            "case_id": normalized_case_id,
            "customer_id": payload.customer_id,
            "order_id": payload.order_id,
            "email_subject": payload.email_subject,
            "email_body": payload.email_body,
            "channel": payload.channel,
            "received_at": datetime.now(UTC).isoformat(),
        }
    }
    return CaseState.model_validate(raw_state)


def _dump_or_none(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value)


def run_case(request_payload: RunCaseRequest, runtime: WebRuntime) -> RunCaseResponse:
    """Execute the graph and map the final state to the web response contract."""

    started_at = perf_counter()
    state = _build_case_state(request_payload)
    final_state = run_graph(state, deps=runtime.deps)
    runtime_ms = int((perf_counter() - started_at) * 1000)

    return RunCaseResponse(
        case_id=final_state.input.case_id,
        triage=_dump_or_none(final_state.triage),
        context=_dump_or_none(final_state.context),
        resolution=_dump_or_none(final_state.resolution),
        finalize=_dump_or_none(final_state.finalize),
        security_events=list(final_state.security_events),
        output_guard_passed=bool(final_state.output_guard_passed),
        runtime_ms=max(runtime_ms, 0),
    )


def _load_json_list(source_path: Path) -> list[dict[str, Any]]:
    with source_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Scenarios file must contain a JSON list: {source_path}")
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Scenario entry at index {index} must be a JSON object: {source_path}")
    return payload


def _preview_from_item(item: dict[str, Any], index: int, source_label: str) -> ScenarioPreview:
    scenario_input = item.get("input")
    if isinstance(scenario_input, dict):
        customer_id = str(scenario_input.get("customer_id", "CUST-1001"))
        order_id = str(scenario_input.get("order_id", "ORD-5001"))
        email_subject = str(scenario_input.get("email_subject", "Customer complaint"))
        email_body = str(scenario_input.get("email_body", ""))
        preferred_language_raw = None
        expected = item.get("expected")
        if isinstance(expected, dict):
            preferred_language_raw = expected.get("response_language")
        preferred_language = str(preferred_language_raw) if preferred_language_raw is not None else None
    else:
        customer_id = str(item.get("customer_id", "CUST-1001"))
        order_id = str(item.get("order_id", "ORD-5001"))
        email_subject = str(item.get("email_subject", "Customer complaint"))
        email_body = str(item.get("email_body", ""))
        preferred_language_raw = item.get("preferred_language")
        preferred_language = str(preferred_language_raw) if preferred_language_raw is not None else None

    return ScenarioPreview(
        id=str(item.get("id", f"{source_label}_scenario_{index}")),
        title=str(item.get("title", f"{source_label} Scenario {index}")),
        customer_id=customer_id,
        order_id=order_id,
        email_subject=email_subject,
        email_body=email_body,
        preferred_language=preferred_language,
    )


def load_scenario_previews(scenarios_file: Path | None = None) -> list[ScenarioPreview]:
    """Load scenarios to power UI prefills."""

    if scenarios_file is not None:
        payload = _load_json_list(scenarios_file)
        return [_preview_from_item(item, index, "custom") for index, item in enumerate(payload, start=1)]

    source_paths = [
        ("playground", _default_scenarios_file()),
        ("eval", _default_eval_scenarios_file()),
    ]
    previews: list[ScenarioPreview] = []
    seen_ids: set[str] = set()

    for source_label, source_path in source_paths:
        if not source_path.exists():
            LOGGER.warning("Scenario source missing for web UI: %s", source_path)
            continue

        payload = _load_json_list(source_path)
        for index, item in enumerate(payload, start=1):
            preview = _preview_from_item(item, index=index, source_label=source_label)
            if preview.id in seen_ids:
                LOGGER.warning("Duplicate scenario id skipped in web UI: %s", preview.id)
                continue
            seen_ids.add(preview.id)
            previews.append(preview)

    if not previews:
        raise ValueError("No scenario previews available from default sources.")
    return previews
