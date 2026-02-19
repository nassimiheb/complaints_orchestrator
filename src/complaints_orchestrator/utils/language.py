"""Language detection and response-language selection helpers."""

from __future__ import annotations

import logging
import re

from complaints_orchestrator.constants import ResponseLanguage

LOGGER = logging.getLogger(__name__)

FRENCH_HINTS = {
    "bonjour",
    "merci",
    "commande",
    "remboursement",
    "defectueux",
    "retard",
    "livraison",
    "echange",
    "escalade",
    "probleme",
}
ENGLISH_HINTS = {
    "hello",
    "thanks",
    "order",
    "refund",
    "defective",
    "delivery",
    "delay",
    "exchange",
    "issue",
    "support",
}
FRENCH_ACCENT_PATTERN = re.compile(r"[àâçéèêëîïôûùüÿœ]")


def _record_event(event: str, security_events: list[str] | None, logger: logging.Logger | None) -> None:
    if security_events is not None:
        security_events.append(event)
    (logger or LOGGER).info("Security event: %s", event)


def _normalize_language(language: str | ResponseLanguage | None) -> ResponseLanguage | None:
    if language is None:
        return None
    if isinstance(language, ResponseLanguage):
        return language
    raw = str(language).strip().upper()
    if raw in {"FR", "FRENCH"}:
        return ResponseLanguage.FR
    if raw in {"EN", "ENGLISH"}:
        return ResponseLanguage.EN
    return None


def detect_language(text: str, default: ResponseLanguage = ResponseLanguage.EN) -> ResponseLanguage:
    tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9']+", text.lower())
    token_set = set(tokens)

    fr_score = len(token_set.intersection(FRENCH_HINTS))
    en_score = len(token_set.intersection(ENGLISH_HINTS))
    if FRENCH_ACCENT_PATTERN.search(text.lower()):
        fr_score += 1

    if fr_score > en_score:
        return ResponseLanguage.FR
    if en_score > fr_score:
        return ResponseLanguage.EN
    return default


def choose_response_language(
    detected_language: str | ResponseLanguage | None,
    preferred_language: str | ResponseLanguage | None = None,
    security_events: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> ResponseLanguage:
    normalized_detected = _normalize_language(detected_language)
    if normalized_detected is not None:
        _record_event(f"LANGUAGE_DETECTED_{normalized_detected.value}", security_events, logger)
        return normalized_detected

    normalized_preferred = _normalize_language(preferred_language)
    if normalized_preferred is not None:
        _record_event("LANGUAGE_FALLBACK_TO_MEMORY", security_events, logger)
        _record_event(f"LANGUAGE_SELECTED_{normalized_preferred.value}", security_events, logger)
        return normalized_preferred

    _record_event("LANGUAGE_DEFAULTED_EN", security_events, logger)
    return ResponseLanguage.EN
