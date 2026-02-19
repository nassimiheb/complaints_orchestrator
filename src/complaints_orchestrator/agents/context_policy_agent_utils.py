"""Utility helpers for context-policy agent data shaping and retrieval prep."""

from __future__ import annotations

from typing import Any

from complaints_orchestrator.constants import ResponseLanguage
from complaints_orchestrator.state import CaseState
from complaints_orchestrator.utils.rag_security import sanitize_rag_text


def to_int(raw: object, default: int = 0) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def to_float(raw: object, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def coerce_confidence(raw: object, field_name: str = "context_confidence") -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name} from Mistral: {raw}") from exc
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return round(value, 2)


def coerce_policy_constraints(raw: object) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("policy_constraints must be a list in Mistral context policy output.")
    constraints: list[str] = []
    for item in raw:
        value = sanitize_rag_text(str(item), max_chars=180)
        if value and value not in constraints:
            constraints.append(value)
    return constraints


def sanitize_customer_context(raw: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {
        "customer_id": str(raw.get("customer_id", "")),
        "preferred_language": str(raw.get("preferred_language", "")).upper(),
        "loyalty_tier": str(raw.get("loyalty_tier", "")).upper(),
        "account_age_days": to_int(raw.get("account_age_days", 0)),
        "lifetime_orders": to_int(raw.get("lifetime_orders", 0)),
        "ninety_day_compensation_total": round(to_float(raw.get("ninety_day_compensation_total", 0.0)), 2),
        "fraud_watch": bool(raw.get("fraud_watch", False)),
    }


def sanitize_order_context(raw: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {
        "order_id": str(raw.get("order_id", "")),
        "currency": str(raw.get("currency", "")).upper(),
        "order_total": round(to_float(raw.get("order_total", 0.0)), 2),
        "item_count": to_int(raw.get("item_count", 0)),
        "status": str(raw.get("status", "")).upper(),
    }


def summarize_case_history(raw: dict[str, Any]) -> dict[str, str | int | float | bool]:
    raw_cases = raw.get("cases")
    cases = raw_cases if isinstance(raw_cases, list) else []
    latest_case = cases[0] if cases else {}
    open_case_count = to_int(raw.get("open_case_count", 0))
    recent_escalations_count = to_int(raw.get("recent_escalations_count", 0))
    total_cases = len(cases)
    repeat_claim_suspected = total_cases >= 2 or recent_escalations_count > 0
    return {
        "customer_id": str(raw.get("customer_id", "")),
        "total_cases": total_cases,
        "open_case_count": open_case_count,
        "recent_escalations_count": recent_escalations_count,
        "latest_case_decision": str(latest_case.get("decision", "")).upper(),
        "latest_case_status": str(latest_case.get("status", "")).upper(),
        "repeat_claim_suspected": repeat_claim_suspected,
    }


def build_rag_query(
    state: CaseState,
    customer_context: dict[str, str | int | float | bool],
    order_context: dict[str, str | int | float | bool],
) -> str:
    triage = state.triage
    assert triage is not None
    parts = [
        f"complaint_type={triage.complaint_type}",
        f"urgency={triage.urgency.value}",
        f"order_status={order_context['status']}",
        f"order_total={order_context['order_total']}",
        f"fraud_watch={customer_context['fraud_watch']}",
        state.input.email_subject,
    ]
    return sanitize_rag_text(" ".join(str(p) for p in parts), max_chars=280)


def retrieve_policy_material(
    *,
    query: str,
    language: ResponseLanguage,
    policy_types: tuple[str, ...],
    retriever: Any,
    top_k_per_policy: int,
) -> list[dict[str, str]]:
    top_k = max(top_k_per_policy, 1)
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for policy_type in policy_types:
        retrieved = retriever.retrieve(
            query=query,
            language=language.value,
            top_k=top_k,
            policy_type=policy_type,
        )
        for row in retrieved:
            doc_id = str(row.get("doc_id", "")).strip()
            snippet = sanitize_rag_text(str(row.get("snippet", "")), max_chars=220)
            if not doc_id or not snippet:
                continue
            key = f"{doc_id}:{snippet}"
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "doc_id": doc_id,
                    "policy_type": str(row.get("policy_type", policy_type)).upper(),
                    "snippet": snippet,
                }
            )
    return output


def build_mistral_payload(
    state: CaseState,
    customer_context: dict[str, str | int | float | bool],
    order_context: dict[str, str | int | float | bool],
    case_history_summary: dict[str, str | int | float | bool],
    policy_material: list[dict[str, str]],
) -> dict[str, Any]:
    triage = state.triage
    assert triage is not None
    return {
        "task": "context_policy_analysis",
        "triage": {
            "complaint_type": triage.complaint_type,
            "urgency": triage.urgency.value,
            "risk_flags": [flag.value for flag in triage.risk_flags],
            "response_language": triage.response_language.value,
        },
        "customer_context": customer_context,
        "order_context": order_context,
        "case_history_summary": case_history_summary,
        "retrieved_policies": policy_material[:8],
    }


def fallback_policy_constraints(policy_material: list[dict[str, str]]) -> list[str]:
    constraints: list[str] = []
    for row in policy_material:
        snippet = row.get("snippet", "").strip()
        if not snippet:
            continue
        sentence = snippet.split(".")[0].strip()
        if sentence and sentence not in constraints:
            constraints.append(sentence if sentence.endswith((".", "!", "?")) else f"{sentence}.")
        if len(constraints) >= 5:
            break
    if not constraints:
        constraints.append("Validate policy eligibility before making any compensation decision.")
    return constraints


def collect_policy_sources_and_snippets(
    policy_material: list[dict[str, str]],
    snippet_cap: int = 6,
) -> tuple[list[str], list[str]]:
    policy_source_ids: list[str] = []
    for row in policy_material:
        doc_id = row["doc_id"]
        if doc_id not in policy_source_ids:
            policy_source_ids.append(doc_id)
    rag_snippets = [row["snippet"] for row in policy_material][:snippet_cap]
    return policy_source_ids, rag_snippets

