"""Utility helpers for resolution agent parsing, confidence, and formatting."""

from __future__ import annotations

from complaints_orchestrator.constants import DecisionType, ResponseLanguage
from complaints_orchestrator.state import CaseState


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return default


def normalize_text(value: object) -> str:
    return str(value).strip().upper().replace(" ", "_")


def canonicalize_complaint_type(value: object) -> str:
    normalized = normalize_text(value)
    aliases = {
        "PRODUCT_DEFECT": "DEFECTIVE_ITEM",
        "DEFECTIVE_PRODUCT": "DEFECTIVE_ITEM",
        "ITEM_DEFECT": "DEFECTIVE_ITEM",
        "DELIVERY_ISSUE": "LATE_DELIVERY",
        "SHIPPING_DELAY": "LATE_DELIVERY",
        "DELIVERY_PROBLEM": "LATE_DELIVERY",
        "TRACKING_ISSUE": "TRACKING_REQUEST",
    }
    return aliases.get(normalized, normalized)


def coerce_confidence(raw: object, field_name: str = "resolution_confidence") -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name} from Mistral: {raw}") from exc
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return round(value, 2)


def contains_any(text: str, options: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(opt in lowered for opt in options)


def pick_best_decision(scores: dict[DecisionType, float]) -> DecisionType:
    tie_break_order = [
        DecisionType.ESCALATE,
        DecisionType.REFUND,
        DecisionType.EXCHANGE,
        DecisionType.VOUCHER,
        DecisionType.INFO_ONLY,
    ]
    best_value = max(scores.values())
    candidates = {decision for decision, value in scores.items() if value == best_value}
    for decision in tie_break_order:
        if decision in candidates:
            return decision
    return DecisionType.INFO_ONLY


def score_to_confidence(
    best_score: float,
    triage_confidence: float,
    context_confidence: float,
) -> float:
    score_component = min(max(best_score / 100.0, 0.0), 1.0)
    blended = (0.6 * score_component) + (0.2 * triage_confidence) + (0.2 * context_confidence)
    return round(min(max(blended, 0.0), 1.0), 2)


def compute_voucher_value(state: CaseState) -> float:
    context = state.context
    assert context is not None
    order_total = to_float(context.order_context.get("order_total", 0.0))
    comp_total_90d = to_float(context.customer_context.get("ninety_day_compensation_total", 0.0))

    value = max(10.0, min(order_total * 0.18, 60.0))
    if comp_total_90d >= 50.0:
        value *= 0.75
    return round(max(value, 5.0), 2)


def make_fallback_email(
    language: ResponseLanguage,
    case_id: str,
    hitl_reason: str | None,
) -> tuple[str, str]:
    if language == ResponseLanguage.FR:
        subject = f"Mise a jour de votre dossier {case_id}"
        body = (
            "Bonjour,\n\n"
            "Merci pour votre message. Votre demande a ete transmise a un specialiste pour revue prioritaire.\n"
            "Nous reviendrons vers vous rapidement avec une resolution claire.\n\n"
            "Cordialement,\nSupport Client"
        )
    else:
        subject = f"Update on your case {case_id}"
        body = (
            "Hello,\n\n"
            "Thank you for your message. Your request has been sent to a specialist for priority review.\n"
            "We will follow up shortly with a clear resolution.\n\n"
            "Best regards,\nCustomer Support"
        )

    if hitl_reason:
        body = f"{body}\n\nReference: manual review required."
    return subject, body


def build_ticket_payload(
    state: CaseState,
    decision: DecisionType,
    hitl_reason: str | None,
) -> dict[str, object]:
    triage = state.triage
    assert triage is not None
    return {
        "case_id": state.input.case_id,
        "customer_id": state.input.customer_id,
        "order_id": state.input.order_id,
        "decision": decision.value,
        "urgency": triage.urgency.value,
        "risk_flags": [flag.value for flag in triage.risk_flags],
        "hitl_reason": hitl_reason or "",
    }
