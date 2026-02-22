""" tests for Agent 1: Triage and routing."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from complaints_orchestrator.agents.triage_agent import TriageSignals, run_triage  # noqa: E402
from complaints_orchestrator.constants import ResponseLanguage, RiskFlag, RouteType  # noqa: E402
from complaints_orchestrator.state import CaseState, TriageOutput  # noqa: E402


def _base_state() -> CaseState:
    return CaseState.model_validate(
        {
            "input": {
                "case_id": "CASE-TRIAGE-1",
                "customer_id": "CUST-1001",
                "order_id": "ORD-5001",
                "email_subject": "Need help",
                "email_body": "My package arrived damaged and I need support.",
                "channel": "EMAIL",
                "received_at": "2026-02-19T12:00:00Z",
            }
        }
    )


def _mistral_response_payload(
    complaint_type: str = "DEFECTIVE_ITEM",
    sentiment: str = "NEGATIVE",
    urgency: str = "HIGH",
    risk_flags: list[str] | None = None,
    triage_plan: str = "Get context and resolve according to policy.",
    triage_confidence: float = 0.88,
) -> dict:
    content = {
        "complaint_type": complaint_type,
        "sentiment": sentiment,
        "urgency": urgency,
        "risk_flags": risk_flags or [],
        "triage_plan": triage_plan,
        "triage_confidence": triage_confidence,
    }
    return {"choices": [{"message": {"content": json.dumps(content)}}]}


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestTriageAgent(unittest.TestCase):
    def test_legal_risk_forces_immediate_escalation(self) -> None:
        state = _base_state()
        state.redacted_email_body = "I will contact legal support."

        payload = _mistral_response_payload(risk_flags=["LEGAL_THREAT"], urgency="CRITICAL")
        with patch(
            "complaints_orchestrator.agents.triage_agent.request.urlopen",
            return_value=_FakeHTTPResponse(payload),
        ):
            run_triage(state, signals=TriageSignals(mistral_api_key="test-key"))

        self.assertIsNotNone(state.triage)
        assert state.triage is not None
        self.assertEqual(state.triage.route_decision, RouteType.ESCALATE_IMMEDIATE)
        self.assertIn(RiskFlag.LEGAL_THREAT, state.triage.risk_flags)

    def test_triage_uses_redacted_email_body_not_raw_body(self) -> None:
        state = _base_state()
        state.input.email_body = "RAW EMAIL MUST NOT BE SENT"
        state.redacted_email_body = "SAFE REDACTED BODY FOR TRIAGE"

        def _mock_urlopen(req, timeout=20):
            body = json.loads(req.data.decode("utf-8"))
            user_payload = json.loads(body["messages"][1]["content"])
            self.assertEqual(user_payload["redacted_email_body"], "SAFE REDACTED BODY FOR TRIAGE")
            self.assertNotIn("RAW EMAIL MUST NOT BE SENT", user_payload["redacted_email_body"])
            return _FakeHTTPResponse(_mistral_response_payload())

        with patch(
            "complaints_orchestrator.agents.triage_agent.request.urlopen",
            side_effect=_mock_urlopen,
        ):
            run_triage(state, signals=TriageSignals(mistral_api_key="test-key"))

        self.assertIsNotNone(state.triage)

    def test_language_fallback_to_memory_for_short_body(self) -> None:
        state = _base_state()
        state.redacted_email_body = "ok"

        with patch(
            "complaints_orchestrator.agents.triage_agent.request.urlopen",
            return_value=_FakeHTTPResponse(_mistral_response_payload()),
        ):
            run_triage(
                state,
                signals=TriageSignals(
                    preferred_language="FR",
                    mistral_api_key="test-key",
                ),
            )

        self.assertIsNotNone(state.triage)
        assert state.triage is not None
        self.assertEqual(state.triage.response_language, ResponseLanguage.FR)
        self.assertIn("LANGUAGE_FALLBACK_TO_MEMORY", state.security_events)

    def test_risk_flags_are_mistral_only_no_augmentation(self) -> None:
        state = _base_state()
        state.redacted_email_body = "This is the third time. I demand refund of 500 EUR."

        with patch(
            "complaints_orchestrator.agents.triage_agent.request.urlopen",
            return_value=_FakeHTTPResponse(_mistral_response_payload(risk_flags=[])),
        ):
            run_triage(state, signals=TriageSignals(mistral_api_key="test-key"))

        self.assertIsNotNone(state.triage)
        assert state.triage is not None
        self.assertEqual(state.triage.risk_flags, [])

    def test_triage_output_contract_matches_state_model(self) -> None:
        state = _base_state()
        state.redacted_email_body = "Hello, I have a delivery delay and need help."

        with patch(
            "complaints_orchestrator.agents.triage_agent.request.urlopen",
            return_value=_FakeHTTPResponse(_mistral_response_payload()),
        ):
            run_triage(state, signals=TriageSignals(mistral_api_key="test-key"))

        self.assertIsNotNone(state.triage)
        assert state.triage is not None
        validated = TriageOutput.model_validate(state.triage.model_dump())
        self.assertGreaterEqual(validated.triage_confidence, 0.0)
        self.assertLessEqual(validated.triage_confidence, 1.0)

    def test_mistral_path_used_when_api_returns_valid_json(self) -> None:
        state = _base_state()
        state.redacted_email_body = "I will post this on social media unless fixed."

        payload = _mistral_response_payload(
            complaint_type="PUBLIC_COMPLAINT",
            urgency="CRITICAL",
            risk_flags=["PUBLIC_EXPOSURE"],
            triage_plan="Escalate immediately to specialist queue.",
            triage_confidence=0.91,
        )
        with patch(
            "complaints_orchestrator.agents.triage_agent.request.urlopen",
            return_value=_FakeHTTPResponse(payload),
        ):
            run_triage(
                state,
                signals=TriageSignals(
                    mistral_api_key="test-key",
                    mistral_model="mistral-small-latest",
                ),
            )

        self.assertIsNotNone(state.triage)
        assert state.triage is not None
        self.assertEqual(state.triage.route_decision, RouteType.ESCALATE_IMMEDIATE)
        self.assertIn(RiskFlag.PUBLIC_EXPOSURE, state.triage.risk_flags)
        self.assertEqual(state.triage.complaint_type, "PUBLIC_COMPLAINT")
        self.assertIn("TRIAGE_MISTRAL_USED", state.security_events)

    def test_mistral_failure_raises_runtime_error(self) -> None:
        state = _base_state()
        state.redacted_email_body = "My order is late, please update me."

        with patch(
            "complaints_orchestrator.agents.triage_agent.request.urlopen",
            side_effect=OSError("network issue"),
        ):
            with self.assertRaises(RuntimeError):
                run_triage(state, signals=TriageSignals(mistral_api_key="test-key"))


if __name__ == "__main__":
    unittest.main()
