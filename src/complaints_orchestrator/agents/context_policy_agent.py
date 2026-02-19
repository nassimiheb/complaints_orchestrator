"""Agent 2: Context and policy enrichment logic."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import request

from complaints_orchestrator.rag.build_index import DEFAULT_COLLECTION_NAME
from complaints_orchestrator.rag.retriever import PolicyRetriever
from complaints_orchestrator.state import CaseState, ContextOutput
from complaints_orchestrator.agents.context_policy_agent_utils import (
    build_mistral_payload,
    build_rag_query,
    coerce_confidence,
    coerce_policy_constraints,
    collect_policy_sources_and_snippets,
    fallback_policy_constraints,
    retrieve_policy_material,
    sanitize_customer_context,
    sanitize_order_context,
    summarize_case_history,
)
from complaints_orchestrator.tools.registry import call_tool
from complaints_orchestrator.utils.mistral import (
    request_chat_json_object,
    resolve_mistral_api_key,
    resolve_mistral_model,
)

LOGGER = logging.getLogger(__name__)
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


def _request_mistral_policy_constraints(
    policy_payload: dict[str, Any],
    signals: ContextPolicySignals,
) -> dict[str, object]:
    api_key = resolve_mistral_api_key(
        signals.mistral_api_key,
        "MISTRAL_API_KEY is required for context policy agent. No fallback is enabled.",
    )
    model = resolve_mistral_model(signals.mistral_model)

    system_prompt = (
        "You are a retail complaints context and policy analyst. "
        "Return strict JSON only with keys: policy_constraints, context_confidence. "
        "policy_constraints must be a list of concise policy/tone constraints for the resolver agent. "
        "context_confidence must be a float between 0 and 1."
    )
    return request_chat_json_object(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_payload=policy_payload,
        timeout_seconds=signals.mistral_timeout_seconds,
        urlopen_fn=request.urlopen,
        network_error_prefix="Mistral context-policy call failed",
        format_error_prefix="Invalid Mistral response format for context policy",
        missing_json_error="Mistral context policy response did not contain a valid JSON object.",
    )


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

    customer_context = sanitize_customer_context(customer_raw)
    order_context = sanitize_order_context(order_raw)
    case_history_summary = summarize_case_history(case_history_raw)
    _record_event("CONTEXT_TOOL_PAYLOAD_MINIMIZED", state)

    rag_query = build_rag_query(state, customer_context=customer_context, order_context=order_context)
    retriever = _resolve_retriever(signals)
    policy_material = retrieve_policy_material(
        query=rag_query,
        language=state.triage.response_language,
        policy_types=POLICY_TYPES,
        retriever=retriever,
        top_k_per_policy=signals.rag_top_k_per_policy,
    )
    _record_event("CONTEXT_RAG_RETRIEVED", state)

    mistral_payload = build_mistral_payload(
        state=state,
        customer_context=customer_context,
        order_context=order_context,
        case_history_summary=case_history_summary,
        policy_material=policy_material,
    )

    _record_event("CONTEXT_MISTRAL_ATTEMPTED", state)
    model_output = _request_mistral_policy_constraints(mistral_payload, signals=signals)
    _record_event("CONTEXT_MISTRAL_USED", state)

    policy_constraints = coerce_policy_constraints(model_output.get("policy_constraints"))
    if not policy_constraints:
        policy_constraints = fallback_policy_constraints(policy_material)
    context_confidence = coerce_confidence(model_output.get("context_confidence"), field_name="context_confidence")

    policy_source_ids, rag_snippets = collect_policy_sources_and_snippets(policy_material, snippet_cap=6)

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
