"""Agent 1: Triage and routing logic (Mistral-required, no local fallback)."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from urllib import error, request

from complaints_orchestrator.constants import (
    ResponseLanguage,
    RiskFlag,
    RouteType,
    SentimentLabel,
    UrgencyLevel,
)
from complaints_orchestrator.state import CaseState, TriageOutput
from complaints_orchestrator.utils.language import choose_response_language, detect_language
from complaints_orchestrator.utils.pii import redact_for_triage

LOGGER = logging.getLogger(__name__)
MISTRAL_CHAT_COMPLETIONS_URL = "https://api.mistral.ai/v1/chat/completions"

@dataclass(frozen=True)
class TriageSignals:
    preferred_language: str | ResponseLanguage | None = None
    mistral_api_key: str | None = None
    mistral_model: str | None = None
    mistral_timeout_seconds: int = 20


def _record_event(event: str, state: CaseState, logger: logging.Logger | None = None) -> None:
    state.security_events.append(event)
    (logger or LOGGER).info("Security event: %s", event)


def _resolve_mistral_api_key(signals: TriageSignals) -> str:
    if signals.mistral_api_key and signals.mistral_api_key.strip():
        return signals.mistral_api_key.strip()
    env_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if env_key:
        return env_key
    raise RuntimeError("MISTRAL_API_KEY is required for triage. No fallback is enabled.")


def _resolve_mistral_model(signals: TriageSignals) -> str:
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


def _request_mistral_triage(text: str, signals: TriageSignals) -> dict[str, object]:
    api_key = _resolve_mistral_api_key(signals)
    model = _resolve_mistral_model(signals)

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
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
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
        raise RuntimeError(f"Mistral triage call failed: {exc}") from exc

    try:
        parsed_response = json.loads(raw_response)
        message_content = parsed_response["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Invalid Mistral response format: {exc}") from exc

    raw_content = _extract_message_text(message_content)
    model_output = _extract_json_object(raw_content)
    if model_output is None:
        raise RuntimeError("Mistral response did not contain a valid JSON object.")
    return model_output


def _coerce_sentiment(raw: object) -> SentimentLabel:
    value = str(raw).strip().upper()
    if value == SentimentLabel.NEGATIVE.value:
        return SentimentLabel.NEGATIVE
    if value == SentimentLabel.POSITIVE.value:
        return SentimentLabel.POSITIVE
    if value == SentimentLabel.NEUTRAL.value:
        return SentimentLabel.NEUTRAL
    raise ValueError(f"Invalid sentiment from Mistral: {raw}")


def _coerce_urgency(raw: object) -> UrgencyLevel:
    value = str(raw).strip().upper()
    for urgency in UrgencyLevel:
        if value == urgency.value:
            return urgency
    raise ValueError(f"Invalid urgency from Mistral: {raw}")


def _coerce_risk_flags(raw: object) -> list[RiskFlag]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("risk_flags must be a list in Mistral triage output.")
    output: list[RiskFlag] = []
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
            LOGGER.warning("Ignoring unknown risk flag from Mistral output: %s", value)
    return output


def _coerce_confidence(raw: object) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid triage_confidence from Mistral: {raw}") from exc
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return round(value, 2)


def _route_for_risk_flags(risk_flags: list[RiskFlag]) -> RouteType:
    if RiskFlag.LEGAL_THREAT in risk_flags or RiskFlag.PUBLIC_EXPOSURE in risk_flags:
        return RouteType.ESCALATE_IMMEDIATE
    return RouteType.NEED_CONTEXT


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

    sentiment = _coerce_sentiment(model_output.get("sentiment"))
    urgency = _coerce_urgency(model_output.get("urgency"))
    risk_flags = _coerce_risk_flags(model_output.get("risk_flags"))

    triage_plan = str(model_output.get("triage_plan", "")).strip()
    if not triage_plan:
        raise ValueError("Mistral triage output must include triage_plan.")

    confidence = _coerce_confidence(model_output.get("triage_confidence"))
    route = _route_for_risk_flags(risk_flags)

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
