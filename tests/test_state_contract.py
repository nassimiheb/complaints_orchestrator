"""Tests showing what CaseState.model_validate() does."""
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from complaints_orchestrator.state import CaseState


def build_minimal_valid_payload() -> dict:
    return {
        "input": {
            "case_id": "CASE-9010",
            "customer_id": "CUST-1001",
            "order_id": "ORD-5001",
            "email_subject": "Package arrived damaged",
            "email_body": "My jacket arrived damaged and I want a refund.",
            "channel": "EMAIL",
            "received_at": "2026-02-18T10:30:00Z",
        },
        "redacted_email_body": "My jacket arrived damaged and I want a refund.",
        "security_events": ["PII_REDACTED"],
        "output_guard_passed": False,
    }


class TestCaseStateValidation(unittest.TestCase):
    def test_valid_payload_is_converted_to_typed_state(self) -> None:
        payload = build_minimal_valid_payload()

        state = CaseState.model_validate(payload)

        self.assertEqual(state.input.case_id, "CASE-9010")
        self.assertEqual(state.security_events, ["PII_REDACTED"])
        self.assertFalse(state.output_guard_passed)

    def test_unknown_fields_are_rejected(self) -> None:
        payload = build_minimal_valid_payload()
        payload["unexpected_field"] = "should fail"

        with self.assertRaises(Exception):
            CaseState.model_validate(payload)

    def test_constraints_are_enforced(self) -> None:
        payload = build_minimal_valid_payload()
        payload["triage"] = {
            "complaint_type": "DEFECTIVE_ITEM",
            "sentiment": "NEGATIVE",
            "urgency": "HIGH",
            "detected_language": "EN",
            "response_language": "EN",
            "risk_flags": [],
            "triage_plan": "Need context and policy retrieval.",
            "route_decision": "NEED_CONTEXT",
            "triage_confidence": 1.5,
        }

        with self.assertRaises(Exception):
            CaseState.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
