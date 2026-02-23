"""LangGraph orchestration for the complaints workflow."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from complaints_orchestrator.agents.context_policy_agent import (
    ContextPolicySignals,
    run_context_policy,
)
from complaints_orchestrator.agents.resolution_agent import ResolutionSignals, run_resolution
from complaints_orchestrator.agents.triage_agent import TriageSignals, run_triage
from complaints_orchestrator.config import AppConfig
from complaints_orchestrator.constants import CaseStatus, DecisionType, RouteType
from complaints_orchestrator.memory.store import MemoryStore
from complaints_orchestrator.rag.build_index import DEFAULT_COLLECTION_NAME
from complaints_orchestrator.state import CaseState, ContextOutput, FinalizeOutput
from complaints_orchestrator.utils.pii import redact_for_triage

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphDependencies:
    memory_store: MemoryStore | None = None
    triage_signals: TriageSignals | None = None
    context_signals: ContextPolicySignals | None = None
    resolution_signals: ResolutionSignals | None = None


def build_dependencies_from_config(
    config: AppConfig,
    memory_store: MemoryStore | None = None,
) -> GraphDependencies:
    store = memory_store or MemoryStore(db_path=config.sqlite_path)
    return GraphDependencies(
        memory_store=store,
        triage_signals=TriageSignals(
            mistral_api_key=config.mistral_api_key,
            mistral_model=config.model_name,
        ),
        context_signals=ContextPolicySignals(
            mistral_api_key=config.mistral_api_key,
            mistral_model=config.model_name,
            chroma_dir=config.chroma_dir,
            rag_collection_name=DEFAULT_COLLECTION_NAME,
            embedding_provider=os.getenv("CCO_EMBEDDING_PROVIDER"),
            embedding_model=config.embedding_model,
        ),
        resolution_signals=ResolutionSignals(
            mistral_api_key=config.mistral_api_key,
            mistral_model=config.model_name,
            hitl_amount_threshold=config.hitl_amount_threshold,
            low_confidence_threshold=config.low_confidence_threshold,
        ),
    )


def _record_event(event: str, state: CaseState) -> None:
    state.security_events.append(event)
    LOGGER.info("Security event: %s", event)


def _read_preferred_language(memory_store: MemoryStore | None, customer_id: str) -> str | None:
    if memory_store is None:
        return None
    return memory_store.get_preferred_language(customer_id=customer_id)


def _read_compensation_total(memory_store: MemoryStore | None, customer_id: str) -> float:
    if memory_store is None:
        return 0.0
    return memory_store.get_ninety_day_compensation_total(customer_id=customer_id)


def ingest_email_node(state: CaseState, deps: GraphDependencies) -> CaseState:
    """Prepare redacted payload and read memory hints."""

    _record_event("INGEST_STARTED", state)
    if not state.redacted_email_body:
        state.redacted_email_body = redact_for_triage(
            state.input.email_body,
            security_events=state.security_events,
            logger=LOGGER,
        )

    preferred_language = _read_preferred_language(deps.memory_store, state.input.customer_id)
    if preferred_language:
        _record_event("INGEST_MEMORY_PREFERRED_LANGUAGE_FOUND", state)
    else:
        _record_event("INGEST_MEMORY_PREFERRED_LANGUAGE_MISSING", state)
    _record_event("INGEST_COMPLETED", state)
    return state


def _build_triage_signals(state: CaseState, deps: GraphDependencies) -> TriageSignals:
    base = deps.triage_signals or TriageSignals()
    preferred_language = base.preferred_language

    memory_language = _read_preferred_language(deps.memory_store, state.input.customer_id)
    if memory_language:
        preferred_language = memory_language

    return TriageSignals(
        preferred_language=preferred_language,
        mistral_api_key=base.mistral_api_key,
        mistral_model=base.mistral_model,
        mistral_timeout_seconds=base.mistral_timeout_seconds,
    )


def triage_router_node(state: CaseState, deps: GraphDependencies) -> CaseState:
    """Run triage and attach route trace for graph branching."""

    triage_signals = _build_triage_signals(state, deps=deps)
    run_triage(state, signals=triage_signals)
    if state.triage is None:
        raise ValueError("Triage node did not produce triage output.")

    if state.triage.route_decision == RouteType.ESCALATE_IMMEDIATE:
        _record_event("GRAPH_ROUTE_ESCALATE_IMMEDIATE", state)
    else:
        _record_event("GRAPH_ROUTE_NEED_CONTEXT", state)
    return state


def context_policy_node(state: CaseState, deps: GraphDependencies) -> CaseState:
    """Run context and policy enrichment."""

    run_context_policy(state, signals=deps.context_signals)
    return state


def _build_escalation_context_stub(state: CaseState, deps: GraphDependencies) -> ContextOutput:
    triage = state.triage
    if triage is None:
        raise ValueError("Triage output is required before context stub creation.")

    compensation_total = _read_compensation_total(deps.memory_store, state.input.customer_id)
    return ContextOutput(
        customer_context={
            "customer_id": state.input.customer_id,
            "preferred_language": triage.response_language.value,
            "loyalty_tier": "UNKNOWN",
            "account_age_days": 0,
            "lifetime_orders": 0,
            "ninety_day_compensation_total": compensation_total,
            "fraud_watch": False,
        },
        order_context={
            "order_id": state.input.order_id,
            "currency": "EUR",
            "order_total": 0.0,
            "item_count": 0,
            "status": "UNKNOWN",
        },
        case_history_summary={
            "customer_id": state.input.customer_id,
            "total_cases": 0,
            "open_case_count": 0,
            "recent_escalations_count": 0,
            "latest_case_decision": "",
            "latest_case_status": "",
            "repeat_claim_suspected": False,
        },
        policy_constraints=[
            "Immediate legal/public-risk cases require specialist human review before any compensation action."
        ],
        policy_source_ids=[],
        rag_snippets=[],
        context_confidence=0.5,
    )


def resolution_node(state: CaseState, deps: GraphDependencies) -> CaseState:
    """Run resolution and customer email generation."""

    if state.triage is None:
        raise ValueError("Triage output is required before resolution.")

    if state.context is None and state.triage.route_decision == RouteType.ESCALATE_IMMEDIATE:
        state.context = _build_escalation_context_stub(state, deps=deps)
        _record_event("GRAPH_ESCALATE_IMMEDIATE_CONTEXT_STUBBED", state)

    run_resolution(state, signals=deps.resolution_signals)
    return state


def _extract_action_amount(state: CaseState, tool_name: str) -> float:
    if state.resolution is None:
        return 0.0
    for action in state.resolution.tool_actions:
        if action.tool_name != tool_name:
            continue
        if action.action_value is None:
            return 0.0
        return round(float(action.action_value), 2)
    return 0.0


def _resolve_case_status(state: CaseState) -> CaseStatus:
    if state.resolution is None:
        raise ValueError("Resolution output is required before finalize.")
    if state.resolution.decision == DecisionType.ESCALATE:
        return CaseStatus.ESCALATED
    if state.resolution.hitl_required:
        return CaseStatus.PENDING_HITL
    return CaseStatus.RESOLVED


def _build_structured_summary(
    state: CaseState,
    status: CaseStatus,
    compensation_value: float,
) -> dict[str, Any]:
    triage = state.triage
    resolution = state.resolution
    if triage is None or resolution is None:
        raise ValueError("Triage and resolution outputs are required for summary building.")

    return {
        "case_id": state.input.case_id,
        "customer_id": state.input.customer_id,
        "complaint_type": triage.complaint_type,
        "sentiment": triage.sentiment.value,
        "urgency": triage.urgency.value,
        "response_language": triage.response_language.value,
        "decision": resolution.decision.value,
        "status": status.value,
        "hitl_required": resolution.hitl_required,
        "hitl_reason": resolution.hitl_reason or "",
        "compensation_value": compensation_value,
        "output_guard_passed": state.output_guard_passed,
        "policy_source_ids": state.context.policy_source_ids if state.context else [],
        "tool_actions": [
            {
                "tool_name": action.tool_name,
                "status": action.status,
                "reference_id": action.reference_id,
            }
            for action in resolution.tool_actions
        ],
    }


def finalize_node(state: CaseState, deps: GraphDependencies) -> CaseState:
    """Persist structured memory updates and close workflow."""

    triage = state.triage
    resolution = state.resolution
    if triage is None:
        raise ValueError("Triage output is required before finalize.")
    if resolution is None:
        raise ValueError("Resolution output is required before finalize.")

    _record_event("FINALIZE_STARTED", state)
    status = _resolve_case_status(state)
    compensation_value = 0.0
    if resolution.decision == DecisionType.REFUND:
        compensation_value = _extract_action_amount(state, tool_name="issue_refund")
    elif resolution.decision == DecisionType.VOUCHER:
        compensation_value = _extract_action_amount(state, tool_name="create_compensation")

    summary_payload = _build_structured_summary(
        state=state,
        status=status,
        compensation_value=compensation_value,
    )

    if deps.memory_store is not None:
        deps.memory_store.record_finalize_update(
            case_id=state.input.case_id,
            customer_id=state.input.customer_id,
            decision=resolution.decision.value,
            status=status.value,
            compensation_value=compensation_value,
            opened_at=state.input.received_at,
            preferred_language=triage.response_language.value,
            summary_payload=summary_payload,
        )
        _record_event("FINALIZE_MEMORY_UPDATED", state)
    else:
        _record_event("FINALIZE_MEMORY_SKIPPED", state)

    case_summary = (
        f"Case {state.input.case_id}: {triage.complaint_type} -> {resolution.decision.value} ({status.value})"
    )
    state.finalize = FinalizeOutput(
        status=status,
        memory_updates={
            "decision": resolution.decision.value,
            "status": status.value,
            "compensation_value": compensation_value,
            "preferred_language": triage.response_language.value,
            "output_guard_passed": state.output_guard_passed,
        },
        case_summary=case_summary,
    )
    _record_event("FINALIZE_COMPLETED", state)
    return state


def _route_after_triage(state: CaseState) -> str:
    if state.triage is None:
        raise ValueError("Triage output must exist before routing.")
    if state.triage.route_decision == RouteType.ESCALATE_IMMEDIATE:
        return "resolution_node"
    return "context_policy_node"


def build_graph(deps: GraphDependencies | None = None):
    """Build and compile the LangGraph workflow."""

    dependencies = deps or GraphDependencies()

    graph = StateGraph(CaseState)
    graph.add_node("ingest_email_node", lambda state: ingest_email_node(state, dependencies))
    graph.add_node("triage_router_node", lambda state: triage_router_node(state, dependencies))
    graph.add_node("context_policy_node", lambda state: context_policy_node(state, dependencies))
    graph.add_node("resolution_node", lambda state: resolution_node(state, dependencies))
    graph.add_node("finalize_node", lambda state: finalize_node(state, dependencies))

    graph.add_edge(START, "ingest_email_node")
    graph.add_edge("ingest_email_node", "triage_router_node")
    graph.add_conditional_edges(
        "triage_router_node",
        _route_after_triage,
        {
            "context_policy_node": "context_policy_node",
            "resolution_node": "resolution_node",
        },
    )
    graph.add_edge("context_policy_node", "resolution_node")
    graph.add_edge("resolution_node", "finalize_node")
    graph.add_edge("finalize_node", END)
    return graph.compile()


def run_graph(state: CaseState, deps: GraphDependencies | None = None) -> CaseState:
    graph = build_graph(deps=deps)
    output = graph.invoke(state)
    if isinstance(output, CaseState):
        return output
    return CaseState.model_validate(output)
