"""Agent 3: Resolution strategist and customer email writer."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib import request

from complaints_orchestrator.constants import DecisionType, RiskFlag, ResponseLanguage, UrgencyLevel
from complaints_orchestrator.agents.resolution_agent_utils import (
    build_ticket_payload,
    coerce_confidence,
    compute_voucher_value,
    contains_any,
    make_fallback_email,
    normalize_text,
    pick_best_decision,
    score_to_confidence,
    to_bool,
    to_float,
    to_int,
)
from complaints_orchestrator.state import CaseState, ResolutionOutput, ToolActionRecord
from complaints_orchestrator.tools.registry import call_tool
from complaints_orchestrator.utils.mistral import (
    request_chat_json_object,
    resolve_mistral_api_key,
    resolve_mistral_model,
)
from complaints_orchestrator.utils.output_guard import apply_output_guard

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolutionSignals:
    mistral_api_key: str | None = None
    mistral_model: str | None = None
    mistral_timeout_seconds: int = 20
    hitl_amount_threshold: float | None = None
    low_confidence_threshold: float | None = None


def _record_event(event: str, state: CaseState, logger: logging.Logger | None = None) -> None:
    state.security_events.append(event)
    (logger or LOGGER).info("Security event: %s", event)


def _resolve_hitl_amount_threshold(signals: ResolutionSignals) -> float:
    if signals.hitl_amount_threshold is not None:
        return float(signals.hitl_amount_threshold)
    return float(os.getenv("CCO_HITL_AMOUNT_THRESHOLD", "150.0"))


def _resolve_low_confidence_threshold(signals: ResolutionSignals) -> float:
    if signals.low_confidence_threshold is not None:
        return float(signals.low_confidence_threshold)
    return float(os.getenv("CCO_LOW_CONFIDENCE_THRESHOLD", "0.55"))


def _request_mistral_resolution(payload: dict[str, Any], signals: ResolutionSignals) -> dict[str, object]:
    api_key = resolve_mistral_api_key(
        signals.mistral_api_key,
        "MISTRAL_API_KEY is required for resolution. No fallback is enabled.",
    )
    model = resolve_mistral_model(signals.mistral_model)

    system_prompt = (
        "You are a customer support resolution strategist and email writer for fashion retail complaints. "
        "Return strict JSON only with keys: rationale, resolution_confidence, response_subject, response_body. "
        "response_body must be in the requested language (FR or EN), concise, empathetic, and action-oriented. "
        "Never include internal scores, policy IDs, raw tool JSON, or internal routing terms."
    )
    return request_chat_json_object(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_payload=payload,
        timeout_seconds=signals.mistral_timeout_seconds,
        urlopen_fn=request.urlopen,
        network_error_prefix="Mistral resolution call failed",
        format_error_prefix="Invalid Mistral response format for resolution",
        missing_json_error="Mistral resolution response did not contain a valid JSON object.",
    )


def _score_options(state: CaseState) -> dict[DecisionType, float]:
    triage = state.triage
    context = state.context
    assert triage is not None
    assert context is not None

    complaint_type = normalize_text(triage.complaint_type)
    order_status = normalize_text(context.order_context.get("status", ""))
    order_total = to_float(context.order_context.get("order_total", 0.0))
    fraud_watch = to_bool(context.customer_context.get("fraud_watch", False))
    comp_total_90d = to_float(context.customer_context.get("ninety_day_compensation_total", 0.0))
    repeat_claim_suspected = to_bool(context.case_history_summary.get("repeat_claim_suspected", False))
    policy_text = " ".join(context.policy_constraints).lower()

    scores: dict[DecisionType, float] = {
        DecisionType.INFO_ONLY: 15.0,
        DecisionType.VOUCHER: 12.0,
        DecisionType.REFUND: 10.0,
        DecisionType.EXCHANGE: 10.0,
        DecisionType.ESCALATE: 6.0,
    }

    if complaint_type in {"DEFECTIVE_ITEM", "DAMAGED_ITEM"}:
        scores[DecisionType.REFUND] += 45.0
        scores[DecisionType.EXCHANGE] += 28.0
        scores[DecisionType.VOUCHER] += 8.0
    if complaint_type in {"WRONG_ITEM", "SIZE_MISMATCH", "MATERIAL_DIFFERENCE"}:
        scores[DecisionType.EXCHANGE] += 45.0
        scores[DecisionType.REFUND] += 22.0
    if complaint_type in {"LATE_DELIVERY", "DELIVERY_DELAY", "TRACKING_REQUEST"}:
        scores[DecisionType.INFO_ONLY] += 30.0
        scores[DecisionType.VOUCHER] += 20.0
    if complaint_type in {"PUBLIC_COMPLAINT", "LEGAL_COMPLAINT"}:
        scores[DecisionType.ESCALATE] += 55.0

    if order_status == "IN_TRANSIT":
        scores[DecisionType.INFO_ONLY] += 15.0
        scores[DecisionType.REFUND] -= 18.0
        scores[DecisionType.EXCHANGE] -= 12.0
    elif order_status != "DELIVERED":
        scores[DecisionType.INFO_ONLY] += 8.0
        scores[DecisionType.REFUND] -= 8.0

    if triage.urgency in {UrgencyLevel.HIGH, UrgencyLevel.CRITICAL}:
        scores[DecisionType.VOUCHER] += 7.0
    if triage.urgency == UrgencyLevel.CRITICAL:
        scores[DecisionType.ESCALATE] += 20.0

    if fraud_watch:
        scores[DecisionType.ESCALATE] += 30.0
        scores[DecisionType.REFUND] -= 14.0
        scores[DecisionType.VOUCHER] -= 12.0

    if repeat_claim_suspected:
        scores[DecisionType.ESCALATE] += 16.0
        scores[DecisionType.VOUCHER] -= 12.0

    if comp_total_90d >= 60.0:
        scores[DecisionType.VOUCHER] -= 16.0
        scores[DecisionType.REFUND] -= 8.0
        scores[DecisionType.ESCALATE] += 8.0

    if order_total >= 200.0:
        scores[DecisionType.ESCALATE] += 8.0

    if contains_any(policy_text, ("refund is allowed", "remboursement est permis", "refund allowed")):
        scores[DecisionType.REFUND] += 10.0
    if contains_any(policy_text, ("exchange", "echange")):
        scores[DecisionType.EXCHANGE] += 8.0
    if contains_any(policy_text, ("compensation", "voucher", "bon")):
        scores[DecisionType.VOUCHER] += 7.0
    if contains_any(policy_text, ("human review", "revue humaine", "specialist", "escalation")):
        scores[DecisionType.ESCALATE] += 12.0

    if RiskFlag.LEGAL_THREAT in triage.risk_flags or RiskFlag.PUBLIC_EXPOSURE in triage.risk_flags:
        scores[DecisionType.ESCALATE] += 100.0
    if RiskFlag.REPEAT_CLAIM in triage.risk_flags:
        scores[DecisionType.ESCALATE] += 18.0
    if RiskFlag.HIGH_AMOUNT_RISK in triage.risk_flags:
        scores[DecisionType.ESCALATE] += 14.0

    for decision in list(scores.keys()):
        scores[decision] = round(max(scores[decision], 0.0), 2)
    return scores


def _evaluate_hitl(
    state: CaseState,
    proposed_decision: DecisionType,
    combined_confidence: float,
    signals: ResolutionSignals,
) -> tuple[bool, list[str]]:
    triage = state.triage
    context = state.context
    assert triage is not None
    assert context is not None

    order_total = to_float(context.order_context.get("order_total", 0.0))
    recent_escalations_count = to_int(context.case_history_summary.get("recent_escalations_count", 0))
    repeat_claim_suspected = to_bool(context.case_history_summary.get("repeat_claim_suspected", False))
    comp_total_90d = to_float(context.customer_context.get("ninety_day_compensation_total", 0.0))
    policy_text = " ".join(context.policy_constraints).lower()

    amount_threshold = _resolve_hitl_amount_threshold(signals)
    low_conf_threshold = _resolve_low_confidence_threshold(signals)

    reasons: list[str] = []
    if RiskFlag.LEGAL_THREAT in triage.risk_flags or RiskFlag.PUBLIC_EXPOSURE in triage.risk_flags:
        reasons.append("LEGAL_OR_PUBLIC_RISK")
    if RiskFlag.HIGH_AMOUNT_RISK in triage.risk_flags or order_total >= amount_threshold:
        reasons.append("HIGH_AMOUNT_RISK")
    if (
        RiskFlag.REPEAT_CLAIM in triage.risk_flags
        or repeat_claim_suspected
        or recent_escalations_count > 0
    ):
        reasons.append("REPETITION_RISK")
    if combined_confidence < low_conf_threshold:
        reasons.append("LOW_CONFIDENCE")
    if contains_any(policy_text, ("human review", "revue humaine", "before execution", "specialiste")):
        reasons.append("POLICY_REVIEW_REQUIRED")
    if proposed_decision in {DecisionType.REFUND, DecisionType.VOUCHER} and comp_total_90d >= 75.0:
        reasons.append("HIGH_RECENT_COMPENSATION_TOTAL")

    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return (len(deduped) > 0, deduped)


def _execute_actions(
    state: CaseState,
    decision: DecisionType,
    hitl_reason: str | None,
) -> list[ToolActionRecord]:
    triage = state.triage
    context = state.context
    assert triage is not None
    assert context is not None

    order_total = to_float(context.order_context.get("order_total", 0.0))
    currency = str(context.order_context.get("currency", "EUR")).upper() or "EUR"
    actions: list[ToolActionRecord] = []

    if decision == DecisionType.REFUND:
        refund = call_tool(
            tool_name="issue_refund",
            role="resolution_node",
            payload={
                "order_id": state.input.order_id,
                "amount": round(max(order_total, 0.0), 2),
                "currency": currency,
            },
        )
        actions.append(
            ToolActionRecord(
                tool_name="issue_refund",
                status=str(refund.get("status", "UNKNOWN")),
                reference_id=str(refund.get("refund_id", "N/A")),
                confirmation_message=(
                    f"Refund issued for order {state.input.order_id} "
                    f"({round(max(order_total, 0.0), 2)} {currency})."
                ),
            )
        )
        return actions

    if decision == DecisionType.VOUCHER:
        voucher_value = compute_voucher_value(state)
        compensation = call_tool(
            tool_name="create_compensation",
            role="resolution_node",
            payload={
                "case_id": state.input.case_id,
                "type": "VOUCHER",
                "value": voucher_value,
                "currency": currency,
            },
        )
        actions.append(
            ToolActionRecord(
                tool_name="create_compensation",
                status=str(compensation.get("status", "UNKNOWN")),
                reference_id=str(compensation.get("compensation_id", "N/A")),
                confirmation_message=f"Voucher created for {voucher_value} {currency}.",
            )
        )
        return actions

    if decision in {DecisionType.EXCHANGE, DecisionType.ESCALATE}:
        priority = triage.urgency.value if triage.urgency in {UrgencyLevel.HIGH, UrgencyLevel.CRITICAL} else "MEDIUM"
        ticket = call_tool(
            tool_name="create_support_ticket",
            role="resolution_node",
            payload={
                "case_payload": build_ticket_payload(state, decision=decision, hitl_reason=hitl_reason),
                "priority": priority,
            },
        )
        actions.append(
            ToolActionRecord(
                tool_name="create_support_ticket",
                status=str(ticket.get("status", "UNKNOWN")),
                reference_id=str(ticket.get("ticket_id", "N/A")),
                confirmation_message=f"Support ticket opened in queue {ticket.get('queue', 'N/A')}.",
            )
        )
    return actions


def _build_mistral_payload(
    state: CaseState,
    decision: DecisionType,
    hitl_required: bool,
    hitl_reason: str | None,
    option_scores: dict[DecisionType, float],
    strategy_confidence: float,
) -> dict[str, Any]:
    triage = state.triage
    context = state.context
    assert triage is not None
    assert context is not None

    score_table = {decision_type.value: score for decision_type, score in option_scores.items()}
    return {
        "task": "resolution_and_email",
        "decision": decision.value,
        "hitl_required": hitl_required,
        "hitl_reason": hitl_reason or "",
        "response_language": triage.response_language.value,
        "case_summary": {
            "case_id": state.input.case_id,
            "complaint_type": triage.complaint_type,
            "urgency": triage.urgency.value,
            "sentiment": triage.sentiment.value,
            "order_status": str(context.order_context.get("status", "")),
            "order_total": to_float(context.order_context.get("order_total", 0.0)),
        },
        "policy_constraints": context.policy_constraints[:8],
        "option_scores": score_table,
        "strategy_confidence": strategy_confidence,
    }


def run_resolution(state: CaseState, signals: ResolutionSignals | None = None) -> CaseState:
    """Run resolution strategy, actions, and guarded customer email output."""

    signals = signals or ResolutionSignals()
    _record_event("RESOLUTION_STARTED", state)

    if state.triage is None:
        raise ValueError("Triage output is required before running resolution agent.")
    if state.context is None:
        raise ValueError("Context output is required before running resolution agent.")

    option_scores = _score_options(state)
    proposed_decision = pick_best_decision(option_scores)
    strategy_confidence = score_to_confidence(
        best_score=option_scores[proposed_decision],
        triage_confidence=state.triage.triage_confidence,
        context_confidence=state.context.context_confidence,
    )

    hitl_required, hitl_reasons = _evaluate_hitl(
        state=state,
        proposed_decision=proposed_decision,
        combined_confidence=strategy_confidence,
        signals=signals,
    )
    hitl_reason = "; ".join(hitl_reasons) if hitl_reasons else None

    decision = DecisionType.ESCALATE if hitl_required else proposed_decision
    _record_event(f"RESOLUTION_DECISION_{decision.value}", state)
    if hitl_required:
        _record_event("RESOLUTION_HITL_REQUIRED", state)

    mistral_payload = _build_mistral_payload(
        state=state,
        decision=decision,
        hitl_required=hitl_required,
        hitl_reason=hitl_reason,
        option_scores=option_scores,
        strategy_confidence=strategy_confidence,
    )
    _record_event("RESOLUTION_MISTRAL_ATTEMPTED", state)
    model_output = _request_mistral_resolution(mistral_payload, signals=signals)
    _record_event("RESOLUTION_MISTRAL_USED", state)

    model_rationale = str(model_output.get("rationale", "")).strip()
    if not model_rationale:
        model_rationale = "Decision selected using complaint context, policy constraints, and risk controls."

    model_confidence = coerce_confidence(model_output.get("resolution_confidence"), field_name="resolution_confidence")
    resolution_confidence = round((strategy_confidence + model_confidence) / 2.0, 2)

    response_subject = str(model_output.get("response_subject", "")).strip()
    response_body = str(model_output.get("response_body", "")).strip()
    if not response_subject or not response_body:
        fallback_subject, fallback_body = make_fallback_email(
            language=state.triage.response_language,
            case_id=state.input.case_id,
            hitl_reason=hitl_reason,
        )
        response_subject = response_subject or fallback_subject
        response_body = response_body or fallback_body

    guard_result = apply_output_guard(
        subject=response_subject,
        body=response_body,
        security_events=state.security_events,
        logger=LOGGER,
        attempt_sanitize=True,
    )
    state.output_guard_passed = guard_result.passed

    if guard_result.passed:
        final_subject = guard_result.sanitized_subject
        final_body = guard_result.sanitized_body
        final_decision = decision
        final_hitl_required = hitl_required
        final_hitl_reason = hitl_reason
    else:
        final_decision = DecisionType.ESCALATE
        final_hitl_required = True
        fallback_reasons = list(hitl_reasons)
        if "OUTPUT_GUARD_FALLBACK" not in fallback_reasons:
            fallback_reasons.append("OUTPUT_GUARD_FALLBACK")
        final_hitl_reason = "; ".join(fallback_reasons)
        final_subject, final_body = make_fallback_email(
            language=state.triage.response_language,
            case_id=state.input.case_id,
            hitl_reason=final_hitl_reason,
        )
        model_rationale = (
            f"{model_rationale} Output guard fallback forced escalation to manual review."
        )
        _record_event("RESOLUTION_GUARD_FALLBACK_TEMPLATE_USED", state)

    if final_decision == DecisionType.ESCALATE and not final_hitl_required:
        final_hitl_required = True
    if final_decision == DecisionType.ESCALATE and not final_hitl_reason:
        final_hitl_reason = "MANUAL_ESCALATION"

    tool_actions = _execute_actions(
        state=state,
        decision=final_decision,
        hitl_reason=final_hitl_reason,
    )

    state.resolution = ResolutionOutput(
        decision=final_decision,
        rationale=model_rationale,
        hitl_required=final_hitl_required,
        hitl_reason=final_hitl_reason,
        tool_actions=tool_actions,
        response_subject=final_subject,
        response_body=final_body,
        resolution_confidence=resolution_confidence,
    )
    _record_event("RESOLUTION_COMPLETED", state)
    return state
