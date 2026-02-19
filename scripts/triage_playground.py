"""Playground script to run Agent 1 triage on multiple sample or custom cases."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from complaints_orchestrator.agents.triage_agent import TriageSignals, run_triage
from complaints_orchestrator.state import CaseState


def load_cases(cases_file: Path) -> list[dict[str, Any]]:
    with cases_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("Cases file must contain a JSON list.")
    return payload


def list_cases(cases: list[dict[str, Any]]) -> None:
    print("Available triage cases:")
    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id", f"case_{index}"))
        title = str(case.get("title", "Untitled"))
        body = str(case.get("email_body", ""))
        preview = body[:80] + ("..." if len(body) > 80 else "")
        print(f"  {index}. {case_id} | {title}")
        print(f"     {preview}")


def resolve_case(cases: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    selector_stripped = selector.strip()
    if selector_stripped.isdigit():
        idx = int(selector_stripped)
        if idx <= 0 or idx > len(cases):
            raise IndexError(f"Case index out of range: {idx}")
        return cases[idx - 1]
    for case in cases:
        if str(case.get("id", "")).strip().lower() == selector_stripped.lower():
            return case
    raise KeyError(f"Case not found: {selector}")


def build_state_from_case(case: dict[str, Any], run_tag: str) -> CaseState:
    payload = {
        "input": {
            "case_id": str(case.get("id", f"CASE-PLAY-{run_tag}")).upper().replace("-", "_"),
            "customer_id": str(case.get("customer_id", "CUST-1001")),
            "order_id": str(case.get("order_id", "ORD-5001")),
            "email_subject": str(case.get("email_subject", "Customer complaint")),
            "email_body": str(case.get("email_body", "")),
            "channel": "EMAIL",
            "received_at": datetime.now(UTC).isoformat(),
        }
    }
    return CaseState.model_validate(payload)


def print_triage_result(state: CaseState) -> None:
    if state.triage is None:
        print("No triage output produced.")
        return
    triage = state.triage
    print("----- TRIAGE RESULT -----")
    print(f"complaint_type      : {triage.complaint_type}")
    print(f"sentiment           : {triage.sentiment.value}")
    print(f"urgency             : {triage.urgency.value}")
    print(f"detected_language   : {triage.detected_language.value}")
    print(f"response_language   : {triage.response_language.value}")
    print(f"risk_flags          : {[flag.value for flag in triage.risk_flags]}")
    print(f"route_decision      : {triage.route_decision.value}")
    print(f"triage_confidence   : {triage.triage_confidence}")
    print(f"triage_plan         : {triage.triage_plan}")
    print(f"redacted_email_body : {state.redacted_email_body}")
    print(f"security_events     : {state.security_events}")
    print("-------------------------")


def run_single_case(
    case: dict[str, Any],
    mistral_api_key: str | None,
    mistral_model: str | None,
    timeout: int,
    preferred_language_override: str | None,
    run_tag: str,
) -> None:
    state = build_state_from_case(case, run_tag=run_tag)
    preferred_language = preferred_language_override or case.get("preferred_language")

    signals = TriageSignals(
        preferred_language=str(preferred_language) if preferred_language else None,
        mistral_api_key=mistral_api_key,
        mistral_model=mistral_model,
        mistral_timeout_seconds=timeout,
    )
    try:
        run_triage(state, signals=signals)
    except Exception as exc:
        print("TRIAGE FAILED")
        print(f"error: {exc}")
        return
    print_triage_result(state)


def interactive_loop(
    cases: list[dict[str, Any]],
    mistral_api_key: str | None,
    mistral_model: str | None,
    timeout: int,
) -> None:
    while True:
        print()
        list_cases(cases)
        print("Enter case number/id to run, or q to quit:")
        user_input = input("> ").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            print("Exiting interactive triage playground.")
            return
        try:
            selected = resolve_case(cases, user_input)
        except Exception as exc:
            print(f"Invalid selection: {exc}")
            continue
        run_single_case(
            case=selected,
            mistral_api_key=mistral_api_key,
            mistral_model=mistral_model,
            timeout=timeout,
            preferred_language_override=None,
            run_tag=str(user_input),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triage playground: run Agent 1 on sample/custom cases.")
    parser.add_argument(
        "--cases-file",
        default=str(PROJECT_ROOT / "data" / "triage_playground_cases.json"),
        help="Path to JSON file containing sample cases.",
    )
    parser.add_argument("--list", action="store_true", help="List available sample cases and exit.")
    parser.add_argument("--case", help="Case selector: case index (1-based) or case id.")
    parser.add_argument("--all", action="store_true", help="Run all sample cases.")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode to pick cases repeatedly.")
    parser.add_argument("--body", help="Custom complaint body (bypasses sample cases).")
    parser.add_argument("--subject", default="Customer complaint", help="Custom complaint subject.")
    parser.add_argument("--customer-id", default="CUST-1001", help="Custom customer id for --body mode.")
    parser.add_argument("--order-id", default="ORD-5001", help="Custom order id for --body mode.")
    parser.add_argument("--preferred-language", help="Preferred language override (FR/EN).")
    parser.add_argument("--mistral-api-key", help="Override Mistral API key (otherwise uses env MISTRAL_API_KEY).")
    parser.add_argument("--mistral-model", help="Override Mistral model (otherwise uses env CCO_MODEL_NAME).")
    parser.add_argument("--timeout", type=int, default=20, help="Mistral request timeout in seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases_path = Path(args.cases_file).resolve()
    cases = load_cases(cases_path)

    if args.list:
        list_cases(cases)
        return 0

    if args.interactive:
        interactive_loop(
            cases=cases,
            mistral_api_key=args.mistral_api_key,
            mistral_model=args.mistral_model,
            timeout=args.timeout,
        )
        return 0

    if args.body:
        custom_case = {
            "id": "custom_case",
            "title": "Custom user case",
            "email_subject": args.subject,
            "email_body": args.body,
            "customer_id": args.customer_id,
            "order_id": args.order_id,
            "preferred_language": args.preferred_language,
        }
        run_single_case(
            case=custom_case,
            mistral_api_key=args.mistral_api_key,
            mistral_model=args.mistral_model,
            timeout=args.timeout,
            preferred_language_override=args.preferred_language,
            run_tag="custom",
        )
        return 0

    if args.all:
        for idx, case in enumerate(cases, start=1):
            print(f"\n=== Running case {idx}: {case.get('id', f'case_{idx}')} ===")
            run_single_case(
                case=case,
                mistral_api_key=args.mistral_api_key,
                mistral_model=args.mistral_model,
                timeout=args.timeout,
                preferred_language_override=args.preferred_language,
                run_tag=str(idx),
            )
        return 0

    selector = args.case or "1"
    selected_case = resolve_case(cases, selector)
    run_single_case(
        case=selected_case,
        mistral_api_key=args.mistral_api_key,
        mistral_model=args.mistral_model,
        timeout=args.timeout,
        preferred_language_override=args.preferred_language,
        run_tag=selector,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

