"""Playground script to run Agent 2 context-policy on sample or custom cases."""

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

from complaints_orchestrator.agents.context_policy_agent import ContextPolicySignals, run_context_policy
from complaints_orchestrator.rag.build_index import DEFAULT_COLLECTION_NAME, build_index
from complaints_orchestrator.state import CaseState

ALLOWED_LANGUAGES = {"FR", "EN"}
ALLOWED_SENTIMENTS = {"NEGATIVE", "NEUTRAL", "POSITIVE"}
ALLOWED_URGENCIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
ALLOWED_ROUTES = {"ESCALATE_IMMEDIATE", "NEED_CONTEXT", "HITL_REVIEW", "READY_TO_FINALIZE"}
ALLOWED_RISK_FLAGS = {"LEGAL_THREAT", "PUBLIC_EXPOSURE", "REPEAT_CLAIM", "HIGH_AMOUNT_RISK"}


def load_cases(cases_file: Path) -> list[dict[str, Any]]:
    with cases_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("Cases file must contain a JSON list.")
    return payload


def list_cases(cases: list[dict[str, Any]]) -> None:
    print("Available context-policy cases:")
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


def _normalize_choice(raw: object, allowed: set[str], default: str) -> str:
    value = str(raw if raw is not None else default).strip().upper()
    return value if value in allowed else default


def _normalize_risk_flags(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [part.strip().upper() for part in raw.split(",")]
    elif isinstance(raw, list):
        items = [str(part).strip().upper() for part in raw]
    else:
        items = [str(raw).strip().upper()]

    output: list[str] = []
    for item in items:
        if item and item in ALLOWED_RISK_FLAGS and item not in output:
            output.append(item)
    return output


def _infer_route_from_risk_flags(risk_flags: list[str], default: str) -> str:
    if "LEGAL_THREAT" in risk_flags or "PUBLIC_EXPOSURE" in risk_flags:
        return "ESCALATE_IMMEDIATE"
    return _normalize_choice(default, ALLOWED_ROUTES, "NEED_CONTEXT")


def build_state_from_case(
    case: dict[str, Any],
    run_tag: str,
    response_language_override: str | None = None,
    complaint_type_override: str | None = None,
    urgency_override: str | None = None,
    sentiment_override: str | None = None,
    risk_flags_override: str | None = None,
) -> CaseState:
    input_payload = {
        "case_id": str(case.get("id", f"CASE-CTX-{run_tag}")).upper().replace("-", "_"),
        "customer_id": str(case.get("customer_id", "CUST-1001")),
        "order_id": str(case.get("order_id", "ORD-5001")),
        "email_subject": str(case.get("email_subject", "Customer complaint")),
        "email_body": str(case.get("email_body", "")),
        "channel": "EMAIL",
        "received_at": datetime.now(UTC).isoformat(),
    }

    response_language = _normalize_choice(
        response_language_override or case.get("response_language") or case.get("preferred_language"),
        ALLOWED_LANGUAGES,
        "EN",
    )
    sentiment = _normalize_choice(sentiment_override or case.get("sentiment"), ALLOWED_SENTIMENTS, "NEGATIVE")
    urgency = _normalize_choice(urgency_override or case.get("urgency"), ALLOWED_URGENCIES, "HIGH")
    risk_flags = _normalize_risk_flags(risk_flags_override if risk_flags_override is not None else case.get("risk_flags"))
    route = _infer_route_from_risk_flags(
        risk_flags=risk_flags,
        default=str(case.get("route_decision", "NEED_CONTEXT")),
    )

    triage_payload = {
        "complaint_type": str(complaint_type_override or case.get("complaint_type", "DEFECTIVE_ITEM")).strip().upper(),
        "sentiment": sentiment,
        "urgency": urgency,
        "detected_language": response_language,
        "response_language": response_language,
        "risk_flags": risk_flags,
        "triage_plan": "Playground triage stub to enable Agent 2 testing.",
        "route_decision": route,
        "triage_confidence": float(case.get("triage_confidence", 0.9)),
    }

    payload = {
        "input": input_payload,
        "triage": triage_payload,
        "redacted_email_body": input_payload["email_body"],
    }
    return CaseState.model_validate(payload)


def print_context_result(state: CaseState) -> None:
    if state.context is None:
        print("No context-policy output produced.")
        return

    context = state.context
    triage = state.triage
    assert triage is not None

    print("----- CONTEXT POLICY RESULT -----")
    print(f"case_id                : {state.input.case_id}")
    print(f"customer_id            : {state.input.customer_id}")
    print(f"order_id               : {state.input.order_id}")
    print(f"response_language      : {triage.response_language.value}")
    print(f"customer_context       : {context.customer_context}")
    print(f"order_context          : {context.order_context}")
    print(f"case_history_summary   : {context.case_history_summary}")
    print(f"policy_source_ids      : {context.policy_source_ids}")
    print(f"policy_constraints     : {context.policy_constraints}")
    print(f"rag_snippets           : {context.rag_snippets}")
    print(f"context_confidence     : {context.context_confidence}")
    print(f"security_events        : {state.security_events}")
    print("-------------------------------")


def maybe_build_rag_index(chroma_dir: str, collection_name: str, skip_index_build: bool) -> None:
    if skip_index_build:
        return

    docs_dir = PROJECT_ROOT / "src" / "complaints_orchestrator" / "rag" / "documents"
    stats = build_index(
        docs_dir=str(docs_dir),
        chroma_dir=chroma_dir,
        collection_name=collection_name,
    )
    print(f"RAG index ready: {stats}")


def run_single_case(
    case: dict[str, Any],
    mistral_api_key: str | None,
    mistral_model: str | None,
    timeout: int,
    chroma_dir: str,
    collection_name: str,
    response_language_override: str | None,
    complaint_type_override: str | None,
    urgency_override: str | None,
    sentiment_override: str | None,
    risk_flags_override: str | None,
    run_tag: str,
) -> None:
    state = build_state_from_case(
        case=case,
        run_tag=run_tag,
        response_language_override=response_language_override,
        complaint_type_override=complaint_type_override,
        urgency_override=urgency_override,
        sentiment_override=sentiment_override,
        risk_flags_override=risk_flags_override,
    )

    signals = ContextPolicySignals(
        mistral_api_key=mistral_api_key,
        mistral_model=mistral_model,
        mistral_timeout_seconds=timeout,
        chroma_dir=chroma_dir,
        rag_collection_name=collection_name,
    )
    try:
        run_context_policy(state, signals=signals)
    except Exception as exc:
        print("CONTEXT POLICY FAILED")
        print(f"error: {exc}")
        return
    print_context_result(state)


def interactive_loop(
    cases: list[dict[str, Any]],
    mistral_api_key: str | None,
    mistral_model: str | None,
    timeout: int,
    chroma_dir: str,
    collection_name: str,
    response_language_override: str | None,
    complaint_type_override: str | None,
    urgency_override: str | None,
    sentiment_override: str | None,
    risk_flags_override: str | None,
) -> None:
    while True:
        print()
        list_cases(cases)
        print("Enter case number/id to run, or q to quit:")
        user_input = input("> ").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            print("Exiting interactive context-policy playground.")
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
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            response_language_override=response_language_override,
            complaint_type_override=complaint_type_override,
            urgency_override=urgency_override,
            sentiment_override=sentiment_override,
            risk_flags_override=risk_flags_override,
            run_tag=user_input,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Context-policy playground: run Agent 2 on sample/custom cases with triage stub."
    )
    parser.add_argument(
        "--cases-file",
        default=str(PROJECT_ROOT / "data" / "context_policy_playground_cases.json"),
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

    parser.add_argument("--response-language", help="Triage stub language override (FR/EN).")
    parser.add_argument("--complaint-type", help="Triage stub complaint_type override.")
    parser.add_argument("--urgency", help="Triage stub urgency override.")
    parser.add_argument("--sentiment", help="Triage stub sentiment override.")
    parser.add_argument("--risk-flags", help="Comma-separated triage stub risk flags.")

    parser.add_argument("--mistral-api-key", help="Override Mistral API key (otherwise uses env MISTRAL_API_KEY).")
    parser.add_argument("--mistral-model", help="Override Mistral model (otherwise uses env CCO_MODEL_NAME).")
    parser.add_argument("--timeout", type=int, default=20, help="Mistral request timeout in seconds.")

    parser.add_argument(
        "--chroma-dir",
        default=str(PROJECT_ROOT / "storage" / "chroma"),
        help="Path to local Chroma directory.",
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name.")
    parser.add_argument(
        "--skip-index-build",
        action="store_true",
        help="Skip rebuilding the RAG index before running cases.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases_path = Path(args.cases_file).resolve()
    cases = load_cases(cases_path)

    if args.list:
        list_cases(cases)
        return 0

    maybe_build_rag_index(
        chroma_dir=str(Path(args.chroma_dir).resolve()),
        collection_name=args.collection,
        skip_index_build=args.skip_index_build,
    )

    if args.interactive:
        interactive_loop(
            cases=cases,
            mistral_api_key=args.mistral_api_key,
            mistral_model=args.mistral_model,
            timeout=args.timeout,
            chroma_dir=str(Path(args.chroma_dir).resolve()),
            collection_name=args.collection,
            response_language_override=args.response_language,
            complaint_type_override=args.complaint_type,
            urgency_override=args.urgency,
            sentiment_override=args.sentiment,
            risk_flags_override=args.risk_flags,
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
            "response_language": args.response_language,
            "complaint_type": args.complaint_type or "DEFECTIVE_ITEM",
            "urgency": args.urgency or "HIGH",
            "sentiment": args.sentiment or "NEGATIVE",
            "risk_flags": _normalize_risk_flags(args.risk_flags),
        }
        run_single_case(
            case=custom_case,
            mistral_api_key=args.mistral_api_key,
            mistral_model=args.mistral_model,
            timeout=args.timeout,
            chroma_dir=str(Path(args.chroma_dir).resolve()),
            collection_name=args.collection,
            response_language_override=args.response_language,
            complaint_type_override=args.complaint_type,
            urgency_override=args.urgency,
            sentiment_override=args.sentiment,
            risk_flags_override=args.risk_flags,
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
                chroma_dir=str(Path(args.chroma_dir).resolve()),
                collection_name=args.collection,
                response_language_override=args.response_language,
                complaint_type_override=args.complaint_type,
                urgency_override=args.urgency,
                sentiment_override=args.sentiment,
                risk_flags_override=args.risk_flags,
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
        chroma_dir=str(Path(args.chroma_dir).resolve()),
        collection_name=args.collection,
        response_language_override=args.response_language,
        complaint_type_override=args.complaint_type,
        urgency_override=args.urgency,
        sentiment_override=args.sentiment,
        risk_flags_override=args.risk_flags,
        run_tag=selector,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

