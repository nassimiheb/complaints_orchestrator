"""Agent 1: Triage and routing logic (Mistral-required, no local fallback)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib import request

from complaints_orchestrator.constants import (
    ResponseLanguage,
)
from complaints_orchestrator.state import CaseState, TriageOutput
from complaints_orchestrator.agents.triage_agent_utils import (
    coerce_confidence,
    coerce_risk_flags,
    coerce_sentiment,
    coerce_urgency,
    route_for_risk_flags,
)
from complaints_orchestrator.utils.language import choose_response_language, detect_language
from complaints_orchestrator.utils.mistral import (
    request_chat_json_object,
    resolve_mistral_api_key,
    resolve_mistral_model,
)
from complaints_orchestrator.utils.pii import redact_for_triage

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class TriageSignals:
    preferred_language: str | ResponseLanguage | None = None
    mistral_api_key: str | None = None
    mistral_model: str | None = None
    mistral_timeout_seconds: int = 20


def _record_event(event: str, state: CaseState, logger: logging.Logger | None = None) -> None:
    state.security_events.append(event)
    (logger or LOGGER).info("Security event: %s", event)


def _request_mistral_triage(text: str, signals: TriageSignals) -> dict[str, object]:
    api_key = resolve_mistral_api_key(
        signals.mistral_api_key,
        "MISTRAL_API_KEY is required for triage. No fallback is enabled.",
    )
    model = resolve_mistral_model(signals.mistral_model)

    system_prompt = (
        "You are a complaint triage classifier. "
        "Return strict JSON only with keys: complaint_type, sentiment, urgency, risk_flags, triage_plan, triage_confidence. "
        "Allowed sentiment: NEGATIVE, NEUTRAL, POSITIVE. "
        "Allowed urgency: LOW, MEDIUM, HIGH, CRITICAL. "
        "Allowed risk flags: LEGAL_THREAT, PUBLIC_EXPOSURE, REPEAT_CLAIM, HIGH_AMOUNT_RISK."
    )
    user_payload = {
        "task": "triage_email",
        "redacted_email_body": text,
        "response_format_note": "strict JSON object only",
    }
    return request_chat_json_object(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_payload=user_payload,
        timeout_seconds=signals.mistral_timeout_seconds,
        urlopen_fn=request.urlopen,
        network_error_prefix="Mistral triage call failed",
        format_error_prefix="Invalid Mistral response format",
        missing_json_error="Mistral response did not contain a valid JSON object.",
    )


def run_triage(state: CaseState, signals: TriageSignals | None = None) -> CaseState:
    """Run triage using Mistral output."""

    signals = signals or TriageSignals()
    _record_event("TRIAGE_STARTED", state)

    if not state.redacted_email_body:
        state.redacted_email_body = redact_for_triage(
            state.input.email_body,
            security_events=state.security_events,
            logger=LOGGER,
        )
    text = state.redacted_email_body

    token_count = len(re.findall(r"[^\W\d_]+", text, flags=re.UNICODE))
    detected_language: ResponseLanguage | None = detect_language(text) if token_count >= 3 else None
    response_language = choose_response_language(
        detected_language=detected_language,
        preferred_language=signals.preferred_language,
        security_events=state.security_events,
        logger=LOGGER,
    )
    detected_language_output = detected_language or response_language

    _record_event("TRIAGE_MISTRAL_ATTEMPTED", state)
    model_output = _request_mistral_triage(text=text, signals=signals)
    _record_event("TRIAGE_MISTRAL_USED", state)

    complaint_type = str(model_output.get("complaint_type", "")).strip().upper().replace(" ", "_")
    if not complaint_type:
        raise ValueError("Mistral triage output must include complaint_type.")

    sentiment = coerce_sentiment(model_output.get("sentiment"))
    urgency = coerce_urgency(model_output.get("urgency"))
    risk_flags = coerce_risk_flags(model_output.get("risk_flags"), logger=LOGGER)

    triage_plan = str(model_output.get("triage_plan", "")).strip()
    if not triage_plan:
        raise ValueError("Mistral triage output must include triage_plan.")

    confidence = coerce_confidence(model_output.get("triage_confidence"), field_name="triage_confidence")
    route = route_for_risk_flags(risk_flags)

    state.triage = TriageOutput(
        complaint_type=complaint_type,
        sentiment=sentiment,
        urgency=urgency,
        detected_language=detected_language_output,
        response_language=response_language,
        risk_flags=risk_flags,
        triage_plan=triage_plan,
        route_decision=route,
        triage_confidence=confidence,
    )

    for flag in risk_flags:
        _record_event(f"TRIAGE_RISK_{flag.value}", state)
    _record_event(f"TRIAGE_ROUTE_{route.value}", state)
    _record_event("TRIAGE_COMPLETED", state)
    return state
