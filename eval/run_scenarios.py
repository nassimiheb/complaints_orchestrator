"""evaluation harness for functional and security smoke checks."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from complaints_orchestrator.config import AppConfig
from complaints_orchestrator.graph import build_dependencies_from_config, run_graph
from complaints_orchestrator.logging_config import configure_logging
from complaints_orchestrator.memory.seed_memory import seed as seed_memory
from complaints_orchestrator.memory.store import FORBIDDEN_RAW_EMAIL_KEYS, MemoryStore
from complaints_orchestrator.rag.build_index import DEFAULT_COLLECTION_NAME, build_index
from complaints_orchestrator.state import CaseState
from complaints_orchestrator.utils.output_guard import evaluate_output_guard

LOGGER = logging.getLogger(__name__)
REDACTION_PLACEHOLDER_PREFIX = "[REDACTED_"


@dataclass(frozen=True)
class EvalScenario:
    id: str
    title: str
    input_payload: dict[str, Any]
    expected: dict[str, Any]


@dataclass
class ScenarioResult:
    scenario_id: str
    title: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    route_decision: str = "N/A"
    response_language: str = "N/A"
    decision: str = "N/A"
    demo_output: dict[str, Any] = field(default_factory=dict)


def _default_scenarios_path() -> Path:
    return PROJECT_ROOT / "eval" / "scenarios.json"


def _default_docs_dir() -> Path:
    return PROJECT_ROOT / "src" / "complaints_orchestrator" / "rag" / "documents"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation scenarios.")
    parser.add_argument(
        "--scenarios-file",
        default=str(_default_scenarios_path()),
        help="Path to eval scenarios JSON file.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to environment file.",
    )
    parser.add_argument(
        "--skip-index-build",
        action="store_true",
        help="Skip rebuilding the RAG index before execution.",
    )
    parser.add_argument(
        "--scenario-id",
        help="Run a single scenario by id.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable detailed demo output and print only assertion summary.",
    )
    return parser.parse_args()


def _load_scenarios(path: Path) -> list[EvalScenario]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Scenarios file must contain a JSON list: {path}")
    scenarios: list[EvalScenario] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each scenario entry must be a JSON object.")
        scenario_id = str(item.get("id", "")).strip()
        title = str(item.get("title", "")).strip()
        input_payload = item.get("input")
        expected = item.get("expected")
        if not scenario_id or not title:
            raise ValueError("Each scenario requires non-empty 'id' and 'title'.")
        if not isinstance(input_payload, dict):
            raise ValueError(f"Scenario '{scenario_id}' must include object field 'input'.")
        if not isinstance(expected, dict):
            raise ValueError(f"Scenario '{scenario_id}' must include object field 'expected'.")
        scenarios.append(
            EvalScenario(
                id=scenario_id,
                title=title,
                input_payload=input_payload,
                expected=expected,
            )
        )
    if not scenarios:
        raise ValueError("No scenarios found in scenarios file.")
    return scenarios


def _filter_scenarios(scenarios: list[EvalScenario], scenario_id: str | None) -> list[EvalScenario]:
    if not scenario_id:
        return scenarios
    for scenario in scenarios:
        if scenario.id == scenario_id:
            return [scenario]
    raise KeyError(f"Scenario id not found: {scenario_id}")


def _build_case_state(scenario: EvalScenario) -> CaseState:
    input_payload = scenario.input_payload
    payload = {
        "input": {
            "case_id": scenario.id.upper().replace("-", "_"),
            "customer_id": str(input_payload.get("customer_id", "CUST-1001")),
            "order_id": str(input_payload.get("order_id", "ORD-5001")),
            "email_subject": str(input_payload.get("email_subject", "Customer complaint")),
            "email_body": str(input_payload.get("email_body", "")),
            "channel": str(input_payload.get("channel", "EMAIL")),
            "received_at": str(input_payload.get("received_at", datetime.now(UTC).isoformat())),
        }
    }
    return CaseState.model_validate(payload)


def _assert_expected_outcomes(state: CaseState, expected: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    if state.triage is None:
        return ["Missing triage output."]
    if state.resolution is None:
        return ["Missing resolution output."]

    route_expected = str(expected.get("route_decision", "")).strip().upper()
    if route_expected and state.triage.route_decision.value != route_expected:
        failures.append(
            f"route_decision expected={route_expected} actual={state.triage.route_decision.value}"
        )

    language_expected = str(expected.get("response_language", "")).strip().upper()
    if language_expected and state.triage.response_language.value != language_expected:
        failures.append(
            f"response_language expected={language_expected} actual={state.triage.response_language.value}"
        )

    expected_decisions = expected.get("decision_any_of", [])
    if expected_decisions:
        normalized = [str(value).strip().upper() for value in expected_decisions]
        if state.resolution.decision.value not in normalized:
            failures.append(
                f"decision expected any of {normalized} actual={state.resolution.decision.value}"
            )

    if "hitl_required" in expected:
        expected_hitl = bool(expected["hitl_required"])
        if state.resolution.hitl_required != expected_hitl:
            failures.append(
                f"hitl_required expected={expected_hitl} actual={state.resolution.hitl_required}"
            )

    return failures


def _assert_email_security(state: CaseState) -> list[str]:
    failures: list[str] = []
    if state.resolution is None:
        return ["Missing resolution output for security checks."]

    subject = state.resolution.response_subject
    body = state.resolution.response_body
    guard_check = evaluate_output_guard(subject=subject, body=body)
    if not guard_check.passed:
        failures.append(f"internal leakage detected in final email: {guard_check.violations}")

    combined = f"{subject}\n{body}".upper()
    if REDACTION_PLACEHOLDER_PREFIX in combined:
        failures.append("redaction placeholder leaked in final customer email.")

    return failures


def _assert_no_raw_email_persistence(db_path: str, case_id: str, raw_email_body: str) -> list[str]:
    failures: list[str] = []

    with sqlite3.connect(db_path) as conn:
        column_rows = conn.execute("PRAGMA table_info(cases_memory)").fetchall()
        columns = {str(row[1]).lower() for row in column_rows}
        forbidden_columns = {key.lower() for key in FORBIDDEN_RAW_EMAIL_KEYS}
        leaked_columns = sorted(columns.intersection(forbidden_columns))
        if leaked_columns:
            failures.append(f"forbidden raw email columns present in cases_memory: {leaked_columns}")

        row = conn.execute(
            "SELECT case_id, customer_id, decision, status, compensation_value, opened_at, updated_at "
            "FROM cases_memory WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        if row is None:
            failures.append(f"cases_memory row missing for case_id={case_id}")
        else:
            serialized_row = " | ".join("" if value is None else str(value) for value in row)
            if raw_email_body and raw_email_body in serialized_row:
                failures.append("raw email body content detected in persisted case row.")

    return failures


def _run_single_scenario(
    scenario: EvalScenario,
    db_path: str,
    config: AppConfig,
    memory_store: MemoryStore,
) -> ScenarioResult:
    state = _build_case_state(scenario)
    result = ScenarioResult(
        scenario_id=scenario.id,
        title=scenario.title,
        passed=False,
    )

    try:
        deps = build_dependencies_from_config(config, memory_store=memory_store)
        final_state = run_graph(state, deps=deps)
    except Exception as exc:
        result.failures.append(f"execution error: {exc}")
        return result

    if final_state.triage is not None:
        result.route_decision = final_state.triage.route_decision.value
        result.response_language = final_state.triage.response_language.value
    if final_state.resolution is not None:
        result.decision = final_state.resolution.decision.value
    result.demo_output = _build_demo_output_payload(final_state)

    failures: list[str] = []
    failures.extend(_assert_expected_outcomes(final_state, scenario.expected))
    failures.extend(_assert_email_security(final_state))
    failures.extend(
        _assert_no_raw_email_persistence(
            db_path=db_path,
            case_id=final_state.input.case_id,
            raw_email_body=final_state.input.email_body,
        )
    )

    result.failures = failures
    result.passed = len(failures) == 0
    return result


def _maybe_build_index(config: AppConfig, skip_index_build: bool) -> None:
    if skip_index_build:
        return
    stats = build_index(
        docs_dir=str(_default_docs_dir()),
        chroma_dir=config.chroma_dir,
        collection_name=DEFAULT_COLLECTION_NAME,
    )
    LOGGER.info("RAG index ready for eval: %s", stats)


def _build_demo_output_payload(state: CaseState) -> dict[str, Any]:
    triage = state.triage
    context = state.context
    resolution = state.resolution
    finalize = state.finalize

    tool_actions: list[str] = []
    if resolution is not None:
        for action in resolution.tool_actions:
            tool_actions.append(
                f"{action.tool_name} | status={action.status} | ref={action.reference_id} | {action.confirmation_message}"
            )

    return {
        "case_summary": {
            "type": triage.complaint_type if triage is not None else "N/A",
            "sentiment": triage.sentiment.value if triage is not None else "N/A",
            "urgency": triage.urgency.value if triage is not None else "N/A",
            "language": triage.response_language.value if triage is not None else "N/A",
            "route_decision": triage.route_decision.value if triage is not None else "N/A",
            "status": finalize.status.value if finalize is not None else "N/A",
            "triage_confidence": triage.triage_confidence if triage is not None else None,
            "context_confidence": context.context_confidence if context is not None else None,
            "resolution_confidence": resolution.resolution_confidence if resolution is not None else None,
        },
        "policy_source_ids": context.policy_source_ids if context is not None else [],
        "decision": resolution.decision.value if resolution is not None else "N/A",
        "rationale": resolution.rationale if resolution is not None else "N/A",
        "hitl_required": resolution.hitl_required if resolution is not None else False,
        "hitl_reason": resolution.hitl_reason if resolution is not None else None,
        "tool_actions": tool_actions,
        "response_subject": resolution.response_subject if resolution is not None else "",
        "response_body": resolution.response_body if resolution is not None else "",
        "security_events": state.security_events,
        "output_guard_passed": state.output_guard_passed,
    }


def _print_multiline_block(text: str, indent: str = "  ") -> None:
    if not text:
        print(f"{indent}(empty)")
        return
    for line in text.splitlines():
        print(f"{indent}{line}")


def _print_demo_output(result: ScenarioResult) -> None:
    payload = result.demo_output
    case_summary = payload.get("case_summary", {})
    print("  Demo output")
    print(
        "  Case summary: "
        f"type={case_summary.get('type', 'N/A')} | "
        f"sentiment={case_summary.get('sentiment', 'N/A')} | "
        f"urgency={case_summary.get('urgency', 'N/A')} | "
        f"language={case_summary.get('language', 'N/A')} | "
        f"route={case_summary.get('route_decision', 'N/A')} | "
        f"status={case_summary.get('status', 'N/A')} | "
        f"triage_conf={case_summary.get('triage_confidence', 'N/A')} | "
        f"context_conf={case_summary.get('context_confidence', 'N/A')} | "
        f"resolution_conf={case_summary.get('resolution_confidence', 'N/A')}"
    )
    print(f"  Retrieved policy sources: {payload.get('policy_source_ids', [])}")
    print(f"  Decision: {payload.get('decision', 'N/A')}")
    print(
        f"  HITL: required={payload.get('hitl_required', False)} "
        f"| reason={payload.get('hitl_reason', None)}"
    )
    print(f"  Rationale: {payload.get('rationale', 'N/A')}")

    actions = payload.get("tool_actions", [])
    print("  Tool actions taken:")
    if not actions:
        print("    none")
    else:
        for action in actions:
            print(f"    {action}")

    print("  Final email subject:")
    _print_multiline_block(str(payload.get("response_subject", "")), indent="    ")
    print("  Final email body:")
    _print_multiline_block(str(payload.get("response_body", "")), indent="    ")
    print(f"  Security output: output_guard_passed={payload.get('output_guard_passed', False)}")
    print(f"  Security events: {payload.get('security_events', [])}")


def _print_results(results: list[ScenarioResult], show_demo_output: bool = True) -> None:
    print("Evaluation results")
    print("=" * 72)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} | {result.scenario_id} | {result.title}")
        print(
            f"  route={result.route_decision} | language={result.response_language} | decision={result.decision}"
        )
        if show_demo_output:
            _print_demo_output(result)
        if result.failures:
            for failure in result.failures:
                print(f"  - {failure}")
        print("-" * 72)
    print("=" * 72)
    passed_count = sum(1 for item in results if item.passed)
    print(f"Summary: {passed_count}/{len(results)} scenarios passed.")


def _cleanup_temp_dir(path: Path) -> None:
    for attempt in range(5):
        try:
            gc.collect()
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt == 4:
                LOGGER.warning("Could not remove temporary eval directory: %s", path)
                return
            time.sleep(0.2 * (attempt + 1))


def run(args: argparse.Namespace) -> int:
    config = AppConfig.from_env(env_file=args.env_file)
    configure_logging(config.log_level)
    if not config.mistral_api_key.strip():
        print("Execution failed: MISTRAL_API_KEY is required for eval scenarios.")
        return 1

    scenarios = _load_scenarios(Path(args.scenarios_file).resolve())
    scenarios = _filter_scenarios(scenarios, args.scenario_id)

    tmp_dir = Path(tempfile.mkdtemp(prefix="cco_eval_"))
    memory_store: MemoryStore | None = None
    results: list[ScenarioResult] = []
    try:
        db_path = str(tmp_dir / "eval_memory.db")
        seed_memory(db_path=db_path)
        memory_store = MemoryStore(db_path=db_path)

        _maybe_build_index(config=config, skip_index_build=args.skip_index_build)

        results = [
            _run_single_scenario(
                scenario=scenario,
                db_path=db_path,
                config=config,
                memory_store=memory_store,
            )
            for scenario in scenarios
        ]
    finally:
        memory_store = None
        _cleanup_temp_dir(tmp_dir)

    _print_results(results, show_demo_output=not args.quiet)
    if all(result.passed for result in results):
        return 0
    return 1


def main() -> int:
    args = parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
