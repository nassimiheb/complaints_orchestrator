"""Agent 2: Context and policy enrichment logic."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error, request

from complaints_orchestrator.constants import ResponseLanguage
from complaints_orchestrator.rag.build_index import DEFAULT_COLLECTION_NAME
from complaints_orchestrator.rag.retriever import PolicyRetriever
from complaints_orchestrator.state import CaseState, ContextOutput
from complaints_orchestrator.tools.registry import call_tool
from complaints_orchestrator.utils.rag_security import sanitize_rag_text

LOGGER = logging.getLogger(__name__)
MISTRAL_CHAT_COMPLETIONS_URL = "https://api.mistral.ai/v1/chat/completions"
POLICY_TYPES = ("REFUND_POLICY", "COMPENSATION_POLICY", "TONE_GUIDANCE")


class RetrieverLike(Protocol):
    def retrieve(
        self,
        query: str,
        language: str,
        top_k: int = 4,
        policy_type: str | None = None,
    ) -> list[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class ContextPolicySignals:
    mistral_api_key: str | None = None
    mistral_model: str | None = None
    mistral_timeout_seconds: int = 20
    chroma_dir: str | None = None
    rag_collection_name: str | None = None
    rag_top_k_per_policy: int = 2
    retriever: RetrieverLike | None = None


def _record_event(event: str, state: CaseState, logger: logging.Logger | None = None) -> None:
    state.security_events.append(event)
    (logger or LOGGER).info("Security event: %s", event)


def _resolve_mistral_api_key(signals: ContextPolicySignals) -> str:
    if signals.mistral_api_key and signals.mistral_api_key.strip():
        return signals.mistral_api_key.strip()
    env_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if env_key:
        return env_key
    raise RuntimeError("MISTRAL_API_KEY is required for context policy agent. No fallback is enabled.")


def _resolve_mistral_model(signals: ContextPolicySignals) -> str:
    if signals.mistral_model and signals.mistral_model.strip():
        return signals.mistral_model.strip()
    return os.getenv("CCO_MODEL_NAME", "mistral-small-latest")


def _extract_message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(json.dumps(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _extract_json_object(raw_text: str) -> dict[str, object] | None:
    text = raw_text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _request_mistral_policy_constraints(
    policy_payload: dict[str, Any],
    signals: ContextPolicySignals,
) -> dict[str, object]:
    api_key = _resolve_mistral_api_key(signals)
    model = _resolve_mistral_model(signals)

    system_prompt = (
        "You are a retail complaints context and policy analyst. "
        "Return strict JSON only with keys: policy_constraints, context_confidence. "
        "policy_constraints must be a list of concise policy/tone constraints for the resolver agent. "
        "context_confidence must be a float between 0 and 1."
    )
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(policy_payload, ensure_ascii=True)},
        ],
        "response_format": {"type": "json_object"},
    }

    req = request.Request(
        url=MISTRAL_CHAT_COMPLETIONS_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=signals.mistral_timeout_seconds) as resp:
            raw_response = resp.read().decode("utf-8")
    except (error.URLError, error.HTTPError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Mistral context-policy call failed: {exc}") from exc

    try:
        parsed_response = json.loads(raw_response)
        message_content = parsed_response["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Invalid Mistral response format for context policy: {exc}") from exc

    raw_content = _extract_message_text(message_content)
    model_output = _extract_json_object(raw_content)
    if model_output is None:
        raise RuntimeError("Mistral context policy response did not contain a valid JSON object.")
    return model_output


def _to_int(raw: object, default: int = 0) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _to_float(raw: object, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _coerce_confidence(raw: object) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid context_confidence from Mistral: {raw}") from exc
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return round(value, 2)


def _coerce_policy_constraints(raw: object) -> list[str]:
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


def _sanitize_customer_context(raw: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {
        "customer_id": str(raw.get("customer_id", "")),
        "preferred_language": str(raw.get("preferred_language", "")).upper(),
        "loyalty_tier": str(raw.get("loyalty_tier", "")).upper(),
        "account_age_days": _to_int(raw.get("account_age_days", 0)),
        "lifetime_orders": _to_int(raw.get("lifetime_orders", 0)),
        "ninety_day_compensation_total": round(_to_float(raw.get("ninety_day_compensation_total", 0.0)), 2),
        "fraud_watch": bool(raw.get("fraud_watch", False)),
    }


def _sanitize_order_context(raw: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {
        "order_id": str(raw.get("order_id", "")),
        "currency": str(raw.get("currency", "")).upper(),
        "order_total": round(_to_float(raw.get("order_total", 0.0)), 2),
        "item_count": _to_int(raw.get("item_count", 0)),
        "status": str(raw.get("status", "")).upper(),
    }


def _summarize_case_history(raw: dict[str, Any]) -> dict[str, str | int | float | bool]:
    raw_cases = raw.get("cases")
    cases = raw_cases if isinstance(raw_cases, list) else []
    latest_case = cases[0] if cases else {}
    open_case_count = _to_int(raw.get("open_case_count", 0))
    recent_escalations_count = _to_int(raw.get("recent_escalations_count", 0))
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


def _resolve_retriever(signals: ContextPolicySignals) -> RetrieverLike:
    if signals.retriever is not None:
        return signals.retriever
    chroma_dir = signals.chroma_dir.strip() if signals.chroma_dir else os.getenv("CCO_CHROMA_DIR", "./storage/chroma")
    collection_name = (
        signals.rag_collection_name.strip()
        if signals.rag_collection_name
        else DEFAULT_COLLECTION_NAME
    )
    return PolicyRetriever(chroma_dir=chroma_dir, collection_name=collection_name)


def _build_rag_query(
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


def _retrieve_policy_material(
    query: str,
    language: ResponseLanguage,
    signals: ContextPolicySignals,
) -> list[dict[str, str]]:
    retriever = _resolve_retriever(signals)
    top_k = max(signals.rag_top_k_per_policy, 1)

    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for policy_type in POLICY_TYPES:
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


def _build_mistral_payload(
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


def _fallback_policy_constraints(policy_material: list[dict[str, str]]) -> list[str]:
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


def run_context_policy(state: CaseState, signals: ContextPolicySignals | None = None) -> CaseState:
    """Run context + policy enrichment using read tools, RAG, and Mistral."""

    signals = signals or ContextPolicySignals()
    _record_event("CONTEXT_POLICY_STARTED", state)

    if state.triage is None:
        raise ValueError("Triage output is required before running context policy agent.")

    customer_raw = call_tool(
        tool_name="get_customer_profile",
        role="context_policy_node",
        payload={"customer_id": state.input.customer_id},
    )
    order_raw = call_tool(
        tool_name="get_order_details",
        role="context_policy_node",
        payload={"order_id": state.input.order_id},
    )
    case_history_raw = call_tool(
        tool_name="get_case_history",
        role="context_policy_node",
        payload={"customer_id": state.input.customer_id},
    )
    _record_event("CONTEXT_TOOLS_FETCHED", state)

    customer_context = _sanitize_customer_context(customer_raw)
    order_context = _sanitize_order_context(order_raw)
    case_history_summary = _summarize_case_history(case_history_raw)
    _record_event("CONTEXT_TOOL_PAYLOAD_MINIMIZED", state)

    rag_query = _build_rag_query(state, customer_context=customer_context, order_context=order_context)
    policy_material = _retrieve_policy_material(
        query=rag_query,
        language=state.triage.response_language,
        signals=signals,
    )
    _record_event("CONTEXT_RAG_RETRIEVED", state)

    mistral_payload = _build_mistral_payload(
        state=state,
        customer_context=customer_context,
        order_context=order_context,
        case_history_summary=case_history_summary,
        policy_material=policy_material,
    )

    _record_event("CONTEXT_MISTRAL_ATTEMPTED", state)
    model_output = _request_mistral_policy_constraints(mistral_payload, signals=signals)
    _record_event("CONTEXT_MISTRAL_USED", state)

    policy_constraints = _coerce_policy_constraints(model_output.get("policy_constraints"))
    if not policy_constraints:
        policy_constraints = _fallback_policy_constraints(policy_material)
    context_confidence = _coerce_confidence(model_output.get("context_confidence"))

    policy_source_ids: list[str] = []
    for row in policy_material:
        doc_id = row["doc_id"]
        if doc_id not in policy_source_ids:
            policy_source_ids.append(doc_id)
    rag_snippets = [row["snippet"] for row in policy_material][:6]

    state.context = ContextOutput(
        customer_context=customer_context,
        order_context=order_context,
        case_history_summary=case_history_summary,
        policy_constraints=policy_constraints,
        policy_source_ids=policy_source_ids,
        rag_snippets=rag_snippets,
        context_confidence=context_confidence,
    )
    _record_event("CONTEXT_POLICY_COMPLETED", state)
    return state

