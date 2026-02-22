"""CLI entrypoint for the complaints orchestrator."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from complaints_orchestrator.config import AppConfig
from complaints_orchestrator.graph import build_dependencies_from_config, run_graph
from complaints_orchestrator.logging_config import configure_logging
from complaints_orchestrator.rag.build_index import DEFAULT_COLLECTION_NAME, build_index
from complaints_orchestrator.state import CaseState
from complaints_orchestrator.utils.language import choose_response_language, detect_language
from complaints_orchestrator.utils.output_guard import apply_output_guard
from complaints_orchestrator.utils.pii import redact_for_triage

LOGGER = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_scenarios_file() -> Path:
    return _project_root() / "data" / "triage_playground_cases.json"


def _default_docs_dir() -> Path:
    return _project_root() / "src" / "complaints_orchestrator" / "rag" / "documents"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="complaints-orchestrator",
        description="Complaints Resolution Orchestrator",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        help="Scenario id to run (1-based index from scenarios file).",
    )
    parser.add_argument(
        "--scenarios-file",
        default=str(_default_scenarios_file()),
        help="Path to scenarios JSON file.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to an optional dotenv file.",
    )
    parser.add_argument(
        "--skip-index-build",
        action="store_true",
        help="Skip rebuilding the local RAG index before execution.",
    )
    parser.add_argument(
        "--demo-security",
        action="store_true",
        help="Run a security utility demo.",
    )
    return parser


def _build_demo_state() -> CaseState:
    payload = {
        "input": {
            "case_id": "CASE-DEMO-SEC-1",
            "customer_id": "CUST-1001",
            "order_id": "ORD-5001",
            "email_subject": "Need urgent refund",
            "email_body": (
                "Bonjour, my email is alice.martin@example.com and phone is +33 6 12 34 56 78. "
                "I want a refund for my defective item."
            ),
            "channel": "EMAIL",
            "received_at": "2026-02-19T10:00:00Z",
        },
    }
    return CaseState.model_validate(payload)


def run_security_demo() -> None:
    state = _build_demo_state()

    state.redacted_email_body = redact_for_triage(
        state.input.email_body,
        security_events=state.security_events,
        logger=LOGGER,
    )
    detected_language = detect_language(state.redacted_email_body)
    response_language = choose_response_language(
        detected_language=detected_language,
        preferred_language="EN",
        security_events=state.security_events,
        logger=LOGGER,
    )

    draft_subject = "Complaint update"
    draft_body = (
        "Here is internal score=0.92 and doc_id=REFUND_POLICY_FR.\n"
        '{"refund_id":"RFD-0001","status":"ISSUED"}\n'
        "We are sorry for this issue and we will help you."
    )
    guard = apply_output_guard(
        subject=draft_subject,
        body=draft_body,
        security_events=state.security_events,
        logger=LOGGER,
        attempt_sanitize=True,
    )
    state.output_guard_passed = guard.passed

    print("security demo output")
    print(f"redacted_email_body: {state.redacted_email_body}")
    print(f"detected_language: {detected_language.value}")
    print(f"response_language: {response_language.value}")
    print(f"output_guard_passed: {state.output_guard_passed}")
    print(f"security_events: {state.security_events}")
    print(f"guarded_email_subject: {guard.sanitized_subject}")
    print(f"guarded_email_body: {guard.sanitized_body}")


def _load_scenarios(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Scenarios file must contain a JSON list: {path}")
    return payload


def _resolve_scenario(scenarios: list[dict[str, Any]], scenario_id: int) -> dict[str, Any]:
    if scenario_id <= 0 or scenario_id > len(scenarios):
        raise IndexError(f"Scenario id out of range: {scenario_id}. Available range: 1..{len(scenarios)}")
    return scenarios[scenario_id - 1]


def _build_state_from_scenario(scenario: dict[str, Any], scenario_id: int) -> CaseState:
    case_id = str(scenario.get("id", f"SCENARIO_{scenario_id}")).strip().upper().replace("-", "_")
    payload = {
        "input": {
            "case_id": case_id,
            "customer_id": str(scenario.get("customer_id", "CUST-1001")),
            "order_id": str(scenario.get("order_id", "ORD-5001")),
            "email_subject": str(scenario.get("email_subject", "Customer complaint")),
            "email_body": str(scenario.get("email_body", "")),
            "channel": "EMAIL",
            "received_at": datetime.now(UTC).isoformat(),
        }
    }
    return CaseState.model_validate(payload)


def _maybe_build_rag_index(config: AppConfig, skip_index_build: bool) -> None:
    if skip_index_build:
        return
    stats = build_index(
        docs_dir=str(_default_docs_dir()),
        chroma_dir=config.chroma_dir,
        collection_name=DEFAULT_COLLECTION_NAME,
    )
    LOGGER.info("RAG index ready: %s", stats)


def _print_runtime_output(state: CaseState) -> None:
    triage = state.triage
    context = state.context
    resolution = state.resolution

    print("Case summary")
    if triage is None:
        print("type: N/A | sentiment: N/A | urgency: N/A | language: N/A")
    else:
        print(
            f"type: {triage.complaint_type} | sentiment: {triage.sentiment.value} | "
            f"urgency: {triage.urgency.value} | language: {triage.response_language.value}"
        )

    print("Retrieved policy sources")
    print(context.policy_source_ids if context is not None else [])

    print("Decision + rationale")
    if resolution is None:
        print("decision: N/A")
        print("rationale: N/A")
    else:
        print(f"decision: {resolution.decision.value}")
        print(f"rationale: {resolution.rationale}")

    print("Tool actions taken")
    if resolution is None or not resolution.tool_actions:
        print("none")
    else:
        for action in resolution.tool_actions:
            print(
                f"{action.tool_name} | status={action.status} | ref={action.reference_id} | "
                f"{action.confirmation_message}"
            )

    print("Final email subject")
    print(resolution.response_subject if resolution is not None else "")
    print("Final email body")
    print(resolution.response_body if resolution is not None else "")

    print("Security output")
    print(f"security_events: {state.security_events}")
    print(f"output_guard_passed: {state.output_guard_passed}")


def run(args: argparse.Namespace) -> int:
    config = AppConfig.from_env(env_file=args.env_file)
    configure_logging(config.log_level)

    LOGGER.info("Bootstrap initialized.")
    if args.demo_security:
        run_security_demo()
        return 0

    if args.scenario is None:
        print("Provide --scenario <id> to run the LangGraph workflow.")
        return 1

    try:
        scenarios = _load_scenarios(Path(args.scenarios_file).resolve())
        scenario = _resolve_scenario(scenarios, args.scenario)
        _maybe_build_rag_index(config=config, skip_index_build=args.skip_index_build)

        state = _build_state_from_scenario(scenario=scenario, scenario_id=args.scenario)
        deps = build_dependencies_from_config(config)
        final_state = run_graph(state, deps=deps)
        _print_runtime_output(final_state)
    except Exception as exc:
        LOGGER.exception("Scenario execution failed.")
        print(f"Execution failed: {exc}")
        return 1
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
