"""Phase 5 tests for security utilities and guardrails."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from complaints_orchestrator.constants import ResponseLanguage  # noqa: E402
from complaints_orchestrator.state import CaseState  # noqa: E402
from complaints_orchestrator.utils.language import choose_response_language, detect_language  # noqa: E402
from complaints_orchestrator.utils.output_guard import apply_output_guard, evaluate_output_guard  # noqa: E402
from complaints_orchestrator.utils.pii import redact_for_triage, redact_pii  # noqa: E402


class TestSecurityUtils(unittest.TestCase):
    def test_redact_for_triage_replaces_pii(self) -> None:
        """Purpose: prove triage input is sanitized and redaction events are emitted."""
        events: list[str] = []
        text = "Contact me at john.doe@example.com or +1 415-555-0101."

        redacted = redact_for_triage(text, security_events=events)

        self.assertNotIn("john.doe@example.com", redacted)
        self.assertNotIn("415-555-0101", redacted)
        self.assertIn("[REDACTED_EMAIL]", redacted)
        self.assertIn("[REDACTED_PHONE]", redacted)
        self.assertIn("PII_REDACTED", events)

    def test_language_detection_and_fallback(self) -> None:
        """Purpose: verify language detection and fallback-to-memory behavior."""
        detected = detect_language("Bonjour, je veux un remboursement.")
        self.assertEqual(detected, ResponseLanguage.FR)

        enum_events: list[str] = []
        selected_from_enum = choose_response_language(
            detected_language=detected,
            preferred_language="EN",
            security_events=enum_events,
        )
        self.assertEqual(selected_from_enum, ResponseLanguage.FR)
        self.assertIn("LANGUAGE_DETECTED_FR", enum_events)
        self.assertNotIn("LANGUAGE_FALLBACK_TO_MEMORY", enum_events)

        events: list[str] = []
        selected = choose_response_language(
            detected_language="UNKNOWN",
            preferred_language="FR",
            security_events=events,
        )
        self.assertEqual(selected, ResponseLanguage.FR)
        self.assertIn("LANGUAGE_FALLBACK_TO_MEMORY", events)

    def test_output_guard_rejects_internal_leakage(self) -> None:
        """Purpose: ensure guard flags internal scores, policy ids, and tool JSON leakage."""
        result = evaluate_output_guard(
            subject="Update",
            body='score=0.91 doc_id=REFUND_POLICY_FR {"refund_id":"RFD-1"}',
        )
        self.assertFalse(result.passed)
        self.assertIn("INTERNAL_SCORES", result.violations)
        self.assertIn("INTERNAL_POLICY_IDS", result.violations)
        self.assertIn("TOOL_JSON_BLOB", result.violations)

    def test_output_guard_can_sanitize(self) -> None:
        """Purpose: confirm guard can sanitize unsafe drafts and mark final pass events."""
        events: list[str] = []
        guarded = apply_output_guard(
            subject="Case update",
            body=(
                "This line is safe.\n"
                "score=0.62 should not be sent.\n"
                '{"refund_id":"RFD-0001"}\n'
                "Another safe line."
            ),
            security_events=events,
            attempt_sanitize=True,
        )
        self.assertTrue(guarded.passed)
        self.assertIn("This line is safe.", guarded.sanitized_body)
        self.assertIn("OUTPUT_GUARD_FAILED", events)
        self.assertIn("OUTPUT_GUARD_SANITIZED", events)
        self.assertIn("OUTPUT_GUARD_PASSED", events)

    def test_security_events_recorded_in_state_and_logs(self) -> None:
        """Purpose: validate security events are persisted in state and emitted in logs."""
        state = CaseState.model_validate(
            {
                "input": {
                    "case_id": "CASE-SEC-001",
                    "customer_id": "CUST-1001",
                    "order_id": "ORD-5001",
                    "email_subject": "Need help",
                    "email_body": "Email alice@example.com and phone +33 6 11 22 33 44",
                    "channel": "EMAIL",
                    "received_at": "2026-02-19T11:00:00Z",
                }
            }
        )

        with self.assertLogs("complaints_orchestrator.utils.pii", level="INFO") as pii_logs:
            state.redacted_email_body = redact_for_triage(
                state.input.email_body,
                security_events=state.security_events,
            )
        self.assertTrue(any("Security event" in line for line in pii_logs.output))

        with self.assertLogs("complaints_orchestrator.utils.output_guard", level="INFO") as guard_logs:
            guarded = apply_output_guard(
                subject="Case update",
                body="doc_id=REFUND_POLICY_FR should be removed.",
                security_events=state.security_events,
                attempt_sanitize=True,
            )
        state.output_guard_passed = guarded.passed

        self.assertTrue(any("Security event" in line for line in guard_logs.output))
        self.assertGreater(len(state.security_events), 0)
        self.assertTrue(state.redacted_email_body)


if __name__ == "__main__":
    unittest.main()
