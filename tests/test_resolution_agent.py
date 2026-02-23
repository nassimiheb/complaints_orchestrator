"""tests for Agent 3: resolution strategist and email writer."""

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

from complaints_orchestrator.agents.resolution_agent import ResolutionSignals, run_resolution  # noqa: E402
from complaints_orchestrator.constants import DecisionType  # noqa: E402
from complaints_orchestrator.state import CaseState, ResolutionOutput  # noqa: E402
from complaints_orchestrator.utils.output_guard import GuardResult  # noqa: E402


def _base_state(
    *,
    complaint_type: str = "DEFECTIVE_ITEM",
    urgency: str = "HIGH",
    response_language: str = "EN",
    risk_flags: list[str] | None = None,
    triage_confidence: float = 0.9,
    context_confidence: float = 0.88,
    order_total: float = 99.0,
    order_status: str = "DELIVERED",
    currency: str = "EUR",
    comp_total_90d: float = 10.0,
    repeat_claim_suspected: bool = False,
    recent_escalations_count: int = 0,
    policy_constraints: list[str] | None = None,
) -> CaseState:
    return CaseState.model_validate(
        {
            "input": {
                "case_id": "CASE-RES-1",
                "customer_id": "CUST-1001",
                "order_id": "ORD-5001",
                "email_subject": "Need help with my order",
                "email_body": "My order has an issue and I need a resolution.",
                "channel": "EMAIL",
                "received_at": "2026-02-19T15:00:00Z",
            },
            "triage": {
                "complaint_type": complaint_type,
                "sentiment": "NEGATIVE",
                "urgency": urgency,
                "detected_language": response_language,
                "response_language": response_language,
                "risk_flags": risk_flags or [],
                "triage_plan": "Get context and resolve safely.",
                "route_decision": "NEED_CONTEXT",
                "triage_confidence": triage_confidence,
            },
            "context": {
                "customer_context": {
                    "customer_id": "CUST-1001",
                    "preferred_language": response_language,
                    "loyalty_tier": "GOLD",
                    "account_age_days": 700,
                    "lifetime_orders": 20,
                    "ninety_day_compensation_total": comp_total_90d,
                    "fraud_watch": False,
                },
                "order_context": {
                    "order_id": "ORD-5001",
                    "currency": currency,
                    "order_total": order_total,
                    "item_count": 2,
                    "status": order_status,
                },
                "case_history_summary": {
                    "customer_id": "CUST-1001",
                    "total_cases": 2,
                    "open_case_count": 0,
                    "recent_escalations_count": recent_escalations_count,
                    "latest_case_decision": "VOUCHER",
                    "latest_case_status": "CLOSED",
                    "repeat_claim_suspected": repeat_claim_suspected,
                },
                "policy_constraints": policy_constraints
                or [
                    "Validate eligibility before refund approval.",
                    "Keep communication concise and clear.",
                ],
                "policy_source_ids": ["REFUND_POLICY_EN", "TONE_GUIDANCE_EN"],
                "rag_snippets": [
                    "If defect is confirmed and policy window is valid, refund is allowed.",
                    "Use concise language and explain next action clearly.",
                ],
                "context_confidence": context_confidence,
            },
            "redacted_email_body": "My order has an issue and I need a resolution.",
        }
    )


def _mistral_response_payload(
    *,
    rationale: str = "Decision selected from context, policy constraints, and risk checks.",
    resolution_confidence: float = 0.86,
    response_subject: str = "Update on your request",
    response_body: str = "Hello, we are sorry for the issue. We are taking action now.",
) -> dict:
    content = {
        "rationale": rationale,
        "resolution_confidence": resolution_confidence,
        "response_subject": response_subject,
        "response_body": response_body,
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


class TestResolutionAgent(unittest.TestCase):
    def test_delivery_issue_alias_maps_to_late_delivery_strategy(self) -> None:
        state = _base_state(
            complaint_type="DELIVERY_ISSUE",
            order_total=78.5,
            order_status="IN_TRANSIT",
            policy_constraints=[
                "Compensation is appropriate for delay inconvenience.",
                "Keep communication concise and clear.",
            ],
        )

        with patch(
            "complaints_orchestrator.agents.resolution_agent.request.urlopen",
            return_value=_FakeHTTPResponse(_mistral_response_payload()),
        ):
            run_resolution(state, signals=ResolutionSignals(mistral_api_key="test-key"))

        self.assertIsNotNone(state.resolution)
        assert state.resolution is not None
        self.assertIn(state.resolution.decision, {DecisionType.INFO_ONLY, DecisionType.VOUCHER})
        self.assertFalse(state.resolution.hitl_required)

    def test_product_defect_alias_maps_to_defective_item_strategy(self) -> None:
        state = _base_state(
            complaint_type="PRODUCT_DEFECT",
            order_total=99.0,
            order_status="DELIVERED",
            policy_constraints=[
                "Refund is allowed when defect is confirmed within policy window.",
                "Keep communication concise and clear.",
            ],
        )

        with patch(
            "complaints_orchestrator.agents.resolution_agent.request.urlopen",
            return_value=_FakeHTTPResponse(_mistral_response_payload()),
        ):
            run_resolution(state, signals=ResolutionSignals(mistral_api_key="test-key"))

        self.assertIsNotNone(state.resolution)
        assert state.resolution is not None
        self.assertEqual(state.resolution.decision, DecisionType.REFUND)
        self.assertFalse(state.resolution.hitl_required)

    def test_refund_decision_and_action_for_defective_item(self) -> None:
        state = _base_state(
            complaint_type="DEFECTIVE_ITEM",
            order_total=99.0,
            order_status="DELIVERED",
            policy_constraints=[
                "Refund is allowed when defect is confirmed within policy window.",
                "Keep communication concise and clear.",
            ],
        )

        with patch(
            "complaints_orchestrator.agents.resolution_agent.request.urlopen",
            return_value=_FakeHTTPResponse(_mistral_response_payload()),
        ):
            run_resolution(state, signals=ResolutionSignals(mistral_api_key="test-key"))

        self.assertIsNotNone(state.resolution)
        assert state.resolution is not None
        validated = ResolutionOutput.model_validate(state.resolution.model_dump())
        self.assertEqual(validated.decision, DecisionType.REFUND)
        self.assertFalse(validated.hitl_required)
        self.assertEqual(len(validated.tool_actions), 1)
        self.assertEqual(validated.tool_actions[0].tool_name, "issue_refund")
        self.assertTrue(state.output_guard_passed)
        self.assertGreaterEqual(validated.resolution_confidence, 0.0)
        self.assertLessEqual(validated.resolution_confidence, 1.0)

    def test_voucher_decision_uses_compensation_tool(self) -> None:
        state = _base_state(
            complaint_type="LATE_DELIVERY",
            order_total=70.0,
            order_status="DELIVERED",
            policy_constraints=[
                "Compensation is appropriate for delay inconvenience.",
                "Keep communication concise and clear.",
            ],
        )

        with patch(
            "complaints_orchestrator.agents.resolution_agent.request.urlopen",
            return_value=_FakeHTTPResponse(_mistral_response_payload()),
        ):
            run_resolution(state, signals=ResolutionSignals(mistral_api_key="test-key"))

        self.assertIsNotNone(state.resolution)
        assert state.resolution is not None
        self.assertEqual(state.resolution.decision, DecisionType.VOUCHER)
        self.assertEqual(len(state.resolution.tool_actions), 1)
        self.assertEqual(state.resolution.tool_actions[0].tool_name, "create_compensation")

    def test_hitl_legal_public_risk_forces_escalation(self) -> None:
        state = _base_state(
            complaint_type="PUBLIC_COMPLAINT",
            risk_flags=["LEGAL_THREAT"],
            order_total=80.0,
            order_status="DELIVERED",
        )

        with patch(
            "complaints_orchestrator.agents.resolution_agent.request.urlopen",
            return_value=_FakeHTTPResponse(_mistral_response_payload()),
        ):
            run_resolution(state, signals=ResolutionSignals(mistral_api_key="test-key"))

        self.assertIsNotNone(state.resolution)
        assert state.resolution is not None
        self.assertEqual(state.resolution.decision, DecisionType.ESCALATE)
        self.assertTrue(state.resolution.hitl_required)
        self.assertIn("LEGAL_OR_PUBLIC_RISK", state.resolution.hitl_reason or "")
        self.assertEqual(state.resolution.tool_actions[0].tool_name, "create_support_ticket")

    def test_hitl_low_confidence_forces_escalation(self) -> None:
        state = _base_state(
            complaint_type="DEFECTIVE_ITEM",
            triage_confidence=0.40,
            context_confidence=0.45,
            order_total=95.0,
        )

        with patch(
            "complaints_orchestrator.agents.resolution_agent.request.urlopen",
            return_value=_FakeHTTPResponse(_mistral_response_payload()),
        ):
            run_resolution(state, signals=ResolutionSignals(mistral_api_key="test-key"))

        self.assertIsNotNone(state.resolution)
        assert state.resolution is not None
        self.assertTrue(state.resolution.hitl_required)
        self.assertIn("LOW_CONFIDENCE", state.resolution.hitl_reason or "")
        self.assertEqual(state.resolution.decision, DecisionType.ESCALATE)

    def test_output_guard_fallback_forces_safe_template_and_escalation(self) -> None:
        state = _base_state(
            complaint_type="DEFECTIVE_ITEM",
            response_language="EN",
            order_total=80.0,
        )

        with patch(
            "complaints_orchestrator.agents.resolution_agent.request.urlopen",
            return_value=_FakeHTTPResponse(_mistral_response_payload()),
        ):
            with patch(
                "complaints_orchestrator.agents.resolution_agent.apply_output_guard",
                return_value=GuardResult(
                    passed=False,
                    violations=["INTERNAL_POLICY_IDS"],
                    sanitized_subject="Unsafe",
                    sanitized_body="Unsafe",
                ),
            ):
                run_resolution(state, signals=ResolutionSignals(mistral_api_key="test-key"))

        self.assertIsNotNone(state.resolution)
        assert state.resolution is not None
        self.assertFalse(state.output_guard_passed)
        self.assertTrue(state.resolution.hitl_required)
        self.assertEqual(state.resolution.decision, DecisionType.ESCALATE)
        self.assertIn("OUTPUT_GUARD_FALLBACK", state.resolution.hitl_reason or "")
        self.assertIn("specialist", state.resolution.response_body.lower())
        self.assertEqual(state.resolution.tool_actions[0].tool_name, "create_support_ticket")

    def test_mistral_failure_raises_runtime_error(self) -> None:
        state = _base_state()
        with patch(
            "complaints_orchestrator.agents.resolution_agent.request.urlopen",
            side_effect=OSError("network issue"),
        ):
            with self.assertRaises(RuntimeError):
                run_resolution(state, signals=ResolutionSignals(mistral_api_key="test-key"))

    def test_internal_case_identifier_in_subject_is_replaced_with_order_id(self) -> None:
        state = _base_state()
        payload = _mistral_response_payload(
            response_subject="Your Refund for Order #WEB_CASE_20260222_192255_385035",
            response_body="We processed your refund.",
        )

        with patch(
            "complaints_orchestrator.agents.resolution_agent.request.urlopen",
            return_value=_FakeHTTPResponse(payload),
        ):
            run_resolution(state, signals=ResolutionSignals(mistral_api_key="test-key"))

        self.assertIsNotNone(state.resolution)
        assert state.resolution is not None
        self.assertIn("ORD-5001", state.resolution.response_subject)
        self.assertNotIn("WEB_CASE_", state.resolution.response_subject.upper())
        self.assertNotIn("CASE-RES-1", state.resolution.response_subject.upper())

    def test_escaped_newlines_in_body_are_preserved_as_line_breaks(self) -> None:
        state = _base_state()
        payload = _mistral_response_payload(
            response_subject="Refund update",
            response_body="Hello,\\n\\nWe have processed your refund.\\nThank you.",
        )

        with patch(
            "complaints_orchestrator.agents.resolution_agent.request.urlopen",
            return_value=_FakeHTTPResponse(payload),
        ):
            run_resolution(state, signals=ResolutionSignals(mistral_api_key="test-key"))

        self.assertIsNotNone(state.resolution)
        assert state.resolution is not None
        self.assertIn("\n\n", state.resolution.response_body)
        self.assertNotIn("\\n", state.resolution.response_body)


if __name__ == "__main__":
    unittest.main()
