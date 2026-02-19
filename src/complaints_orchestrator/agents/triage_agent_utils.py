"""Utility helpers for triage agent normalization and routing."""

from __future__ import annotations

import logging

from complaints_orchestrator.constants import RiskFlag, RouteType, SentimentLabel, UrgencyLevel

LOGGER = logging.getLogger(__name__)


def coerce_sentiment(raw: object) -> SentimentLabel:
    value = str(raw).strip().upper()
    if value == SentimentLabel.NEGATIVE.value:
        return SentimentLabel.NEGATIVE
    if value == SentimentLabel.POSITIVE.value:
        return SentimentLabel.POSITIVE
    if value == SentimentLabel.NEUTRAL.value:
        return SentimentLabel.NEUTRAL
    raise ValueError(f"Invalid sentiment from Mistral: {raw}")


def coerce_urgency(raw: object) -> UrgencyLevel:
    value = str(raw).strip().upper()
    for urgency in UrgencyLevel:
        if value == urgency.value:
            return urgency
    raise ValueError(f"Invalid urgency from Mistral: {raw}")


def coerce_risk_flags(raw: object, logger: logging.Logger | None = None) -> list[RiskFlag]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("risk_flags must be a list in Mistral triage output.")
    output: list[RiskFlag] = []
    target_logger = logger or LOGGER
    for item in raw:
        value = str(item).strip().upper()
        matched = False
        for risk in RiskFlag:
            if value == risk.value:
                if risk not in output:
                    output.append(risk)
                matched = True
                break
        if not matched:
            target_logger.warning("Ignoring unknown risk flag from Mistral output: %s", value)
    return output


def coerce_confidence(raw: object, field_name: str = "triage_confidence") -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name} from Mistral: {raw}") from exc
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return round(value, 2)


def route_for_risk_flags(risk_flags: list[RiskFlag]) -> RouteType:
    if RiskFlag.LEGAL_THREAT in risk_flags or RiskFlag.PUBLIC_EXPOSURE in risk_flags:
        return RouteType.ESCALATE_IMMEDIATE
    return RouteType.NEED_CONTEXT

