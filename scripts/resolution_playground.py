"""Playground script to run Agent 3 resolution on sample or custom cases."""

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

from complaints_orchestrator.agents.resolution_agent import ResolutionSignals, run_resolution
from complaints_orchestrator.state import CaseState

ALLOWED_LANGUAGES = {"FR", "EN"}
ALLOWED_SENTIMENTS = {"NEGATIVE", "NEUTRAL", "POSITIVE"}
ALLOWED_URGENCIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
ALLOWED_RISK_FLAGS = {"LEGAL_THREAT", "PUBLIC_EXPOSURE", "REPEAT_CLAIM", "HIGH_AMOUNT_RISK"}
ALLOWED_ROUTE_DECISIONS = {"ESCALATE_IMMEDIATE", "NEED_CONTEXT", "HITL_REVIEW", "READY_TO_FINALIZE"}


def load_cases(cases_file: Path) -> list[dict[str, Any]]:
    with cases_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("Cases file must contain a JSON list.")
    return payload


def list_cases(cases: list[dict[str, Any]]) -> None:
    print("Available resolution cases:")
    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id", f"case_{index}"))
        title = str(case.get("title", "Untitled"))
        complaint_type = str(case.get("complaint_type", "UNKNOWN"))
        preview = str(case.get("email_body", ""))[:80]
        if len(str(case.get("email_body", ""))) > 80:
            preview += "..."
        print(f"  {index}. {case_id} | {title}")
        print(f"     complaint={complaint_type} | {preview}")


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


def _to_float(raw: object, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _to_int(raw: object, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _to_bool(raw: object, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def _normalize_risk_flags(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [part.strip().upper() for part in raw.split(",")]
    elif isinstance(raw, list):
        values = [str(item).strip().upper() for item in raw]
    else:
        values = [str(raw).strip().upper()]

    output: list[str] = []
    for value in values:
        if value and value in ALLOWED_RISK_FLAGS and value not in output:
            output.append(value)
    return output


def _infer_route(risk_flags: list[str], raw_route: object) -> str:
    if "LEGAL_THREAT" in risk_flags or "PUBLIC_EXPOSURE" in risk_flags:
        return "ESCALATE_IMMEDIATE"
    return _normalize_choice(raw_route, ALLOWED_ROUTE_DECISIONS, "NEED_CONTEXT")


def _normalize_policy_constraints(raw: object, language: str) -> list[str]:
    if isinstance(raw, list):
        values = [str(item).strip() for item in raw if str(item).strip()]
    elif isinstance(raw, str) and raw.strip():
        values = [raw.strip()]
    else:
        values = []
    if values:
        return values
    if language == "FR":
        return [
            "Verifier l eligibilite selon la politique avant de finaliser la decision.",
            "Garder une communication claire, concise et orientee solution.",
        ]
    return [
        "Validate eligibility against policy before finalizing the decision.",
        "Keep communication clear, concise, and solution focused.",
    ]


def _default_policy_sources(language: str) -> list[str]:
    return [f"REFUND_POLICY_{language}", f"TONE_GUIDANCE_{language}"]


def _default_rag_snippets(language: str) -> list[str]:
    if language == "FR":
        return [
            "Si le defaut est confirme et dans la fenetre autorisee, le remboursement est permis.",
            "La reponse client doit rester claire et orientee resolution.",
        ]
    return [
        "If defect is confirmed and policy window applies, refund is allowed.",
        "Customer communication must stay clear and solution focused.",
    ]


def build_state_from_case(
    case: dict[str, Any],
    run_tag: str,
    response_language_override: str | None = None,
    complaint_type_override: str | None = None,
    urgency_override: str | None = None,
    sentiment_override: str | None = None,
    risk_flags_override: str | None = None,
    order_total_override: float | None = None,
    order_status_override: str | None = None,
    fraud_watch_override: str | None = None,
    repeat_claim_override: str | None = None,
    recent_escalations_override: int | None = None,
    context_confidence_override: float | None = None,
    triage_confidence_override: float | None = None,
    compensation_total_override: float | None = None,
) -> CaseState:
    input_payload = {
        "case_id": str(case.get("id", f"CASE-RES-{run_tag}")).upper().replace("-", "_"),
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
    route_decision = _infer_route(risk_flags, case.get("route_decision", "NEED_CONTEXT"))
    triage_confidence = _to_float(
        triage_confidence_override if triage_confidence_override is not None else case.get("triage_confidence"),
        0.9,
    )

    order_total = _to_float(order_total_override if order_total_override is not None else case.get("order_total"), 100.0)
    order_status = str(order_status_override if order_status_override is not None else case.get("order_status", "DELIVERED")).strip().upper()
    currency = str(case.get("currency", "EUR")).strip().upper() or "EUR"
    compensation_total = _to_float(
        compensation_total_override if compensation_total_override is not None else case.get("ninety_day_compensation_total"),
        10.0,
    )
    fraud_watch = _to_bool(
        fraud_watch_override if fraud_watch_override is not None else case.get("fraud_watch"),
        False,
    )
    repeat_claim = _to_bool(
        repeat_claim_override if repeat_claim_override is not None else case.get("repeat_claim_suspected"),
        False,
    )
    recent_escalations_count = _to_int(
        recent_escalations_override if recent_escalations_override is not None else case.get("recent_escalations_count"),
        0,
    )
    context_confidence = _to_float(
        context_confidence_override if context_confidence_override is not None else case.get("context_confidence"),
        0.85,
    )

    policy_constraints = _normalize_policy_constraints(case.get("policy_constraints"), response_language)
    policy_source_ids = case.get("policy_source_ids")
    if not isinstance(policy_source_ids, list):
        policy_source_ids = _default_policy_sources(response_language)
    rag_snippets = case.get("rag_snippets")
    if not isinstance(rag_snippets, list):
        rag_snippets = _default_rag_snippets(response_language)

    triage_payload = {
        "complaint_type": str(complaint_type_override or case.get("complaint_type", "DEFECTIVE_ITEM")).strip().upper(),
        "sentiment": sentiment,
        "urgency": urgency,
        "detected_language": response_language,
        "response_language": response_language,
        "risk_flags": risk_flags,
        "triage_plan": "Playground triage stub for resolution testing.",
        "route_decision": route_decision,
        "triage_confidence": triage_confidence,
    }

    context_payload = {
        "customer_context": {
            "customer_id": input_payload["customer_id"],
            "preferred_language": response_language,
            "loyalty_tier": str(case.get("loyalty_tier", "STANDARD")).strip().upper(),
            "account_age_days": _to_int(case.get("account_age_days"), 300),
            "lifetime_orders": _to_int(case.get("lifetime_orders"), 10),
            "ninety_day_compensation_total": compensation_total,
            "fraud_watch": fraud_watch,
        },
        "order_context": {
            "order_id": input_payload["order_id"],
            "currency": currency,
            "order_total": order_total,
            "item_count": _to_int(case.get("item_count"), 1),
            "status": order_status,
        },
        "case_history_summary": {
            "customer_id": input_payload["customer_id"],
            "total_cases": _to_int(case.get("total_cases"), 2),
            "open_case_count": _to_int(case.get("open_case_count"), 0),
            "recent_escalations_count": recent_escalations_count,
            "latest_case_decision": str(case.get("latest_case_decision", "VOUCHER")).strip().upper(),
            "latest_case_status": str(case.get("latest_case_status", "CLOSED")).strip().upper(),
            "repeat_claim_suspected": repeat_claim,
        },
        "policy_constraints": policy_constraints,
        "policy_source_ids": [str(x) for x in policy_source_ids if str(x).strip()],
        "rag_snippets": [str(x) for x in rag_snippets if str(x).strip()],
        "context_confidence": context_confidence,
    }

    payload = {
        "input": input_payload,
        "triage": triage_payload,
        "context": context_payload,
        "redacted_email_body": input_payload["email_body"],
    }
    return CaseState.model_validate(payload)


def print_resolution_result(state: CaseState) -> None:
    if state.resolution is None:
        print("No resolution output produced.")
        return

    resolution = state.resolution
    triage = state.triage
    context = state.context
    assert triage is not None
    assert context is not None

    print("----- RESOLUTION RESULT -----")
    print(f"case_id              : {state.input.case_id}")
    print(f"complaint_type       : {triage.complaint_type}")
    print(f"urgency              : {triage.urgency.value}")
    print(f"response_language    : {triage.response_language.value}")
    print(f"order_total          : {context.order_context.get('order_total')} {context.order_context.get('currency')}")
    print(f"decision             : {resolution.decision.value}")
    print(f"hitl_required        : {resolution.hitl_required}")
    print(f"hitl_reason          : {resolution.hitl_reason}")
    print(f"resolution_confidence: {resolution.resolution_confidence}")
    print(f"rationale            : {resolution.rationale}")
    print("tool_actions         :")
    if not resolution.tool_actions:
        print("  - none")
    for action in resolution.tool_actions:
        print(f"  - {action.tool_name} | status={action.status} | ref={action.reference_id}")
        print(f"    {action.confirmation_message}")
    print(f"output_guard_passed  : {state.output_guard_passed}")
    print(f"response_subject     : {resolution.response_subject}")
    print("response_body        :")
    print(resolution.response_body)
    print(f"security_events      : {state.security_events}")
    print("----------------------------")


def run_single_case(
    case: dict[str, Any],
    mistral_api_key: str | None,
    mistral_model: str | None,
    timeout: int,
    hitl_amount_threshold: float | None,
    low_confidence_threshold: float | None,
    response_language_override: str | None,
    complaint_type_override: str | None,
    urgency_override: str | None,
    sentiment_override: str | None,
    risk_flags_override: str | None,
    order_total_override: float | None,
    order_status_override: str | None,
    fraud_watch_override: str | None,
    repeat_claim_override: str | None,
    recent_escalations_override: int | None,
    context_confidence_override: float | None,
    triage_confidence_override: float | None,
    compensation_total_override: float | None,
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
        order_total_override=order_total_override,
        order_status_override=order_status_override,
        fraud_watch_override=fraud_watch_override,
        repeat_claim_override=repeat_claim_override,
        recent_escalations_override=recent_escalations_override,
        context_confidence_override=context_confidence_override,
        triage_confidence_override=triage_confidence_override,
        compensation_total_override=compensation_total_override,
    )

    signals = ResolutionSignals(
        mistral_api_key=mistral_api_key,
        mistral_model=mistral_model,
        mistral_timeout_seconds=timeout,
        hitl_amount_threshold=hitl_amount_threshold,
        low_confidence_threshold=low_confidence_threshold,
    )
    try:
        run_resolution(state, signals=signals)
    except Exception as exc:
        print("RESOLUTION FAILED")
        print(f"error: {exc}")
        return
    print_resolution_result(state)


def interactive_loop(
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    while True:
        print()
        list_cases(cases)
        print("Enter case number/id to run, or q to quit:")
        user_input = input("> ").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            print("Exiting interactive resolution playground.")
            return
        try:
            selected = resolve_case(cases, user_input)
        except Exception as exc:
            print(f"Invalid selection: {exc}")
            continue

        run_single_case(
            case=selected,
            mistral_api_key=args.mistral_api_key,
            mistral_model=args.mistral_model,
            timeout=args.timeout,
            hitl_amount_threshold=args.hitl_amount_threshold,
            low_confidence_threshold=args.low_confidence_threshold,
            response_language_override=args.response_language,
            complaint_type_override=args.complaint_type,
            urgency_override=args.urgency,
            sentiment_override=args.sentiment,
            risk_flags_override=args.risk_flags,
            order_total_override=args.order_total,
            order_status_override=args.order_status,
            fraud_watch_override=args.fraud_watch,
            repeat_claim_override=args.repeat_claim,
            recent_escalations_override=args.recent_escalations,
            context_confidence_override=args.context_confidence,
            triage_confidence_override=args.triage_confidence,
            compensation_total_override=args.comp_total_90d,
            run_tag=user_input,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolution playground: run Agent 3 on sample/custom cases with triage+context stubs."
    )
    parser.add_argument(
        "--cases-file",
        default=str(PROJECT_ROOT / "data" / "resolution_playground_cases.json"),
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
    parser.add_argument("--order-total", type=float, help="Order total override for context stub.")
    parser.add_argument("--order-status", help="Order status override for context stub.")
    parser.add_argument("--fraud-watch", help="Fraud watch override (true/false).")
    parser.add_argument("--repeat-claim", help="Repeat claim override (true/false).")
    parser.add_argument("--recent-escalations", type=int, help="Recent escalations override.")
    parser.add_argument("--context-confidence", type=float, help="Context confidence override (0-1).")
    parser.add_argument("--triage-confidence", type=float, help="Triage confidence override (0-1).")
    parser.add_argument("--comp-total-90d", type=float, help="90-day compensation total override.")

    parser.add_argument("--mistral-api-key", help="Override Mistral API key (otherwise uses env MISTRAL_API_KEY).")
    parser.add_argument("--mistral-model", help="Override Mistral model (otherwise uses env CCO_MODEL_NAME).")
    parser.add_argument("--timeout", type=int, default=20, help="Mistral request timeout in seconds.")
    parser.add_argument("--hitl-amount-threshold", type=float, help="Override HITL amount threshold.")
    parser.add_argument("--low-confidence-threshold", type=float, help="Override low confidence threshold.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases_path = Path(args.cases_file).resolve()
    cases = load_cases(cases_path)

    if args.list:
        list_cases(cases)
        return 0

    if args.interactive:
        interactive_loop(cases=cases, args=args)
        return 0

    if args.body:
        custom_case = {
            "id": "custom_case",
            "title": "Custom user case",
            "email_subject": args.subject,
            "email_body": args.body,
            "customer_id": args.customer_id,
            "order_id": args.order_id,
            "response_language": args.response_language or "EN",
            "complaint_type": args.complaint_type or "DEFECTIVE_ITEM",
            "urgency": args.urgency or "HIGH",
            "sentiment": args.sentiment or "NEGATIVE",
            "risk_flags": _normalize_risk_flags(args.risk_flags),
            "order_total": args.order_total if args.order_total is not None else 100.0,
            "order_status": args.order_status or "DELIVERED",
        }
        run_single_case(
            case=custom_case,
            mistral_api_key=args.mistral_api_key,
            mistral_model=args.mistral_model,
            timeout=args.timeout,
            hitl_amount_threshold=args.hitl_amount_threshold,
            low_confidence_threshold=args.low_confidence_threshold,
            response_language_override=args.response_language,
            complaint_type_override=args.complaint_type,
            urgency_override=args.urgency,
            sentiment_override=args.sentiment,
            risk_flags_override=args.risk_flags,
            order_total_override=args.order_total,
            order_status_override=args.order_status,
            fraud_watch_override=args.fraud_watch,
            repeat_claim_override=args.repeat_claim,
            recent_escalations_override=args.recent_escalations,
            context_confidence_override=args.context_confidence,
            triage_confidence_override=args.triage_confidence,
            compensation_total_override=args.comp_total_90d,
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
                hitl_amount_threshold=args.hitl_amount_threshold,
                low_confidence_threshold=args.low_confidence_threshold,
                response_language_override=args.response_language,
                complaint_type_override=args.complaint_type,
                urgency_override=args.urgency,
                sentiment_override=args.sentiment,
                risk_flags_override=args.risk_flags,
                order_total_override=args.order_total,
                order_status_override=args.order_status,
                fraud_watch_override=args.fraud_watch,
                repeat_claim_override=args.repeat_claim,
                recent_escalations_override=args.recent_escalations,
                context_confidence_override=args.context_confidence,
                triage_confidence_override=args.triage_confidence,
                compensation_total_override=args.comp_total_90d,
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
        hitl_amount_threshold=args.hitl_amount_threshold,
        low_confidence_threshold=args.low_confidence_threshold,
        response_language_override=args.response_language,
        complaint_type_override=args.complaint_type,
        urgency_override=args.urgency,
        sentiment_override=args.sentiment,
        risk_flags_override=args.risk_flags,
        order_total_override=args.order_total,
        order_status_override=args.order_status,
        fraud_watch_override=args.fraud_watch,
        repeat_claim_override=args.repeat_claim,
        recent_escalations_override=args.recent_escalations,
        context_confidence_override=args.context_confidence,
        triage_confidence_override=args.triage_confidence,
        compensation_total_override=args.comp_total_90d,
        run_tag=selector,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

