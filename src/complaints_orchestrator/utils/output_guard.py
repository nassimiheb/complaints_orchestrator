"""Output guard to block customer-facing leakage of internal details."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

VIOLATION_PATTERNS: dict[str, re.Pattern[str]] = {
    "INTERNAL_SCORES": re.compile(r"\b(score|confidence|triage_confidence|context_confidence|resolution_confidence)\b", re.IGNORECASE),
    "INTERNAL_POLICY_IDS": re.compile(r"\b(doc_id|policy_id|policy_type)\b", re.IGNORECASE),
    "RAW_RAG_EXCERPT": re.compile(r"\b(rag_snippet|source_path|chunk_index)\b", re.IGNORECASE),
    "TOOL_JSON_BLOB": re.compile(r"\{[^{}]{0,600}:[^{}]{0,600}\}", re.IGNORECASE),
}


@dataclass(frozen=True)
class GuardResult:
    passed: bool
    violations: list[str]
    sanitized_subject: str
    sanitized_body: str


def _record_event(event: str, security_events: list[str] | None, logger: logging.Logger | None) -> None:
    if security_events is not None:
        security_events.append(event)
    (logger or LOGGER).info("Security event: %s", event)


def _find_violations(subject: str, body: str) -> list[str]:
    combined = f"{subject}\n{body}"
    violations: list[str] = []
    for name, pattern in VIOLATION_PATTERNS.items():
        if pattern.search(combined):
            violations.append(name)
    return violations


def sanitize_customer_email(subject: str, body: str) -> tuple[str, str]:
    cleaned_subject = subject
    for pattern in VIOLATION_PATTERNS.values():
        cleaned_subject = pattern.sub("[REDACTED_INTERNAL]", cleaned_subject)
    cleaned_subject = re.sub(r"\s+", " ", cleaned_subject).strip()

    kept_lines: list[str] = []
    for line in body.splitlines():
        if any(pattern.search(line) for pattern in VIOLATION_PATTERNS.values()):
            continue
        kept_lines.append(line)
    cleaned_body = "\n".join(kept_lines).strip()
    cleaned_body = re.sub(r"\n{3,}", "\n\n", cleaned_body)
    return cleaned_subject, cleaned_body


def evaluate_output_guard(subject: str, body: str) -> GuardResult:
    violations = _find_violations(subject=subject, body=body)
    return GuardResult(
        passed=len(violations) == 0,
        violations=violations,
        sanitized_subject=subject,
        sanitized_body=body,
    )


def apply_output_guard(
    subject: str,
    body: str,
    security_events: list[str] | None = None,
    logger: logging.Logger | None = None,
    attempt_sanitize: bool = True,
) -> GuardResult:
    initial = evaluate_output_guard(subject=subject, body=body)
    if initial.passed:
        _record_event("OUTPUT_GUARD_PASSED", security_events, logger)
        return initial

    _record_event("OUTPUT_GUARD_FAILED", security_events, logger)
    for violation in initial.violations:
        _record_event(f"OUTPUT_GUARD_{violation}", security_events, logger)

    if not attempt_sanitize:
        return initial

    sanitized_subject, sanitized_body = sanitize_customer_email(subject=subject, body=body)
    after_sanitize = evaluate_output_guard(subject=sanitized_subject, body=sanitized_body)
    if after_sanitize.passed:
        _record_event("OUTPUT_GUARD_SANITIZED", security_events, logger)
        _record_event("OUTPUT_GUARD_PASSED", security_events, logger)
        return GuardResult(
            passed=True,
            violations=initial.violations,
            sanitized_subject=sanitized_subject,
            sanitized_body=sanitized_body,
        )

    _record_event("OUTPUT_GUARD_FALLBACK_REQUIRED", security_events, logger)
    return GuardResult(
        passed=False,
        violations=after_sanitize.violations,
        sanitized_subject=sanitized_subject,
        sanitized_body=sanitized_body,
    )

