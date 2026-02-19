"""PII redaction utilities for safe triage inputs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}\b")
IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){13,19}\b")


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    redaction_count: int
    redacted_entities: list[str]


def _record_event(event: str, security_events: list[str] | None, logger: logging.Logger | None) -> None:
    if security_events is not None:
        security_events.append(event)
    (logger or LOGGER).info("Security event: %s", event)


def redact_pii(
    text: str,
    security_events: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> RedactionResult:
    redaction_count = 0
    redacted_entities: list[str] = []
    redacted_text = text

    def _replace(pattern: re.Pattern[str], replacement: str, entity_name: str) -> None:
        nonlocal redaction_count, redacted_text, redacted_entities
        redacted_text, count = pattern.subn(replacement, redacted_text)
        if count > 0:
            redaction_count += count
            redacted_entities.append(entity_name)

    _replace(EMAIL_PATTERN, "[REDACTED_EMAIL]", "EMAIL")
    _replace(PHONE_PATTERN, "[REDACTED_PHONE]", "PHONE")
    _replace(IBAN_PATTERN, "[REDACTED_IBAN]", "IBAN")
    _replace(CARD_PATTERN, "[REDACTED_CARD]", "CARD")

    if redaction_count > 0:
        unique_entities = sorted(set(redacted_entities))
        _record_event("PII_REDACTED", security_events, logger)
        for entity in unique_entities:
            _record_event(f"PII_{entity}_REDACTED", security_events, logger)
    else:
        _record_event("PII_REDACTION_NOT_NEEDED", security_events, logger)

    return RedactionResult(
        redacted_text=redacted_text,
        redaction_count=redaction_count,
        redacted_entities=sorted(set(redacted_entities)),
    )


def redact_for_triage(
    raw_email_body: str,
    security_events: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> str:
    result = redact_pii(raw_email_body, security_events=security_events, logger=logger)
    return result.redacted_text

