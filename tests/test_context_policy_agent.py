"""Phase 7 tests for Agent 2: context and policy enrichment."""

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

from complaints_orchestrator.agents.context_policy_agent import (  # noqa: E402
    ContextPolicySignals,
    run_context_policy,
)
from complaints_orchestrator.state import CaseState, ContextOutput  # noqa: E402


def _base_state(response_language: str = "FR") -> CaseState:
    return CaseState.model_validate(
        {
            "input": {
                "case_id": "CASE-CONTEXT-1",
                "customer_id": "CUST-1001",
                "order_id": "ORD-5001",
                "email_subject": "Produit defectueux",
                "email_body": "Bonjour, je veux un remboursement pour un article defectueux.",
                "channel": "EMAIL",
                "received_at": "2026-02-19T13:00:00Z",
            },
            "triage": {
                "complaint_type": "DEFECTIVE_ITEM",
                "sentiment": "NEGATIVE",
                "urgency": "HIGH",
                "detected_language": response_language,
                "response_language": response_language,
                "risk_flags": [],
                "triage_plan": "Gather context and policy constraints.",
                "route_decision": "NEED_CONTEXT",
                "triage_confidence": 0.88,
            },
            "redacted_email_body": "Bonjour, je veux un remboursement pour un article defectueux.",
        }
    )


def _mistral_response_payload(
    policy_constraints: list[str] | None = None,
    context_confidence: float = 0.84,
) -> dict:
    content = {
        "policy_constraints": policy_constraints
        or [
            "Validate eligibility before refund approval.",
            "Keep customer communication concise and solution-focused.",
        ],
        "context_confidence": context_confidence,
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


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve(
        self,
        query: str,
        language: str,
        top_k: int = 4,
        policy_type: str | None = None,
    ) -> list[dict[str, object]]:
        self.calls.append(
            {
                "query": query,
                "language": language,
                "top_k": top_k,
                "policy_type": policy_type,
            }
        )
        resolved_policy_type = policy_type or "REFUND_POLICY"
        doc_id = f"{resolved_policy_type}_{language}"
        return [
            {
                "doc_id": doc_id,
                "language": language,
                "policy_type": resolved_policy_type,
                "source_path": f"{resolved_policy_type.lower()}_{language.lower()}.md",
                "snippet": f"{resolved_policy_type} guidance for {language} customer communication.",
                "score": 0.9,
            }
        ]


class TestContextPolicyAgent(unittest.TestCase):
    def test_context_output_contract_matches_state_model(self) -> None:
        state = _base_state(response_language="FR")
        retriever = _FakeRetriever()

        with patch(
            "complaints_orchestrator.agents.context_policy_agent.request.urlopen",
            return_value=_FakeHTTPResponse(_mistral_response_payload()),
        ):
            run_context_policy(
                state,
                signals=ContextPolicySignals(
                    mistral_api_key="test-key",
                    retriever=retriever,
                ),
            )

        self.assertIsNotNone(state.context)
        assert state.context is not None
        validated = ContextOutput.model_validate(state.context.model_dump())
        self.assertGreater(len(validated.policy_constraints), 0)
        self.assertIn("REFUND_POLICY_FR", validated.policy_source_ids)
        self.assertIn("CONTEXT_POLICY_COMPLETED", state.security_events)
        self.assertGreaterEqual(validated.context_confidence, 0.0)
        self.assertLessEqual(validated.context_confidence, 1.0)

    def test_model_input_uses_sanitized_and_minimized_tool_payloads(self) -> None:
        state = _base_state(response_language="EN")
        retriever = _FakeRetriever()

        def _mock_call_tool(tool_name: str, role: str, payload: dict) -> dict:
            self.assertEqual(role, "context_policy_node")
            if tool_name == "get_customer_profile":
                return {
                    "customer_id": "CUST-X",
                    "preferred_language": "EN",
                    "loyalty_tier": "GOLD",
                    "account_age_days": 200,
                    "lifetime_orders": 12,
                    "ninety_day_compensation_total": 40.0,
                    "fraud_watch": False,
                    "email": "private@example.com",
                    "address": {"line1": "secret"},
                }
            if tool_name == "get_order_details":
                return {
                    "order_id": "ORD-X",
                    "customer_id": "CUST-X",
                    "currency": "USD",
                    "order_total": 110.5,
                    "item_count": 2,
                    "status": "DELIVERED",
                    "payment_card": "4111-xxxx-xxxx-1111",
                }
            if tool_name == "get_case_history":
                return {
                    "customer_id": "CUST-X",
                    "open_case_count": 1,
                    "recent_escalations_count": 0,
                    "cases": [{"decision": "VOUCHER", "status": "OPEN"}],
                    "raw_blob": {"sensitive": "dont-send"},
                }
            raise AssertionError(f"Unexpected tool call: {tool_name} {payload}")

        def _mock_urlopen(req, timeout=20):
            body = json.loads(req.data.decode("utf-8"))
            user_payload = json.loads(body["messages"][1]["content"])

            customer_context = user_payload["customer_context"]
            order_context = user_payload["order_context"]
            case_summary = user_payload["case_history_summary"]

            self.assertNotIn("email", customer_context)
            self.assertNotIn("address", customer_context)
            self.assertNotIn("customer_id", order_context)
            self.assertNotIn("payment_card", order_context)
            self.assertNotIn("cases", case_summary)
            self.assertNotIn("raw_blob", case_summary)
            self.assertIn("repeat_claim_suspected", case_summary)
            return _FakeHTTPResponse(_mistral_response_payload())

        with patch(
            "complaints_orchestrator.agents.context_policy_agent.call_tool",
            side_effect=_mock_call_tool,
        ):
            with patch(
                "complaints_orchestrator.agents.context_policy_agent.request.urlopen",
                side_effect=_mock_urlopen,
            ):
                run_context_policy(
                    state,
                    signals=ContextPolicySignals(
                        mistral_api_key="test-key",
                        retriever=retriever,
                    ),
                )

        self.assertIsNotNone(state.context)
        self.assertIn("CONTEXT_TOOL_PAYLOAD_MINIMIZED", state.security_events)

    def test_retrieval_is_language_filtered_by_triage_response_language(self) -> None:
        state = _base_state(response_language="EN")
        retriever = _FakeRetriever()

        with patch(
            "complaints_orchestrator.agents.context_policy_agent.request.urlopen",
            return_value=_FakeHTTPResponse(_mistral_response_payload()),
        ):
            run_context_policy(
                state,
                signals=ContextPolicySignals(
                    mistral_api_key="test-key",
                    retriever=retriever,
                ),
            )

        self.assertEqual(len(retriever.calls), 3)
        self.assertEqual(
            sorted([str(call["policy_type"]) for call in retriever.calls]),
            ["COMPENSATION_POLICY", "REFUND_POLICY", "TONE_GUIDANCE"],
        )
        for call in retriever.calls:
            self.assertEqual(call["language"], "EN")

    def test_mistral_failure_raises_runtime_error(self) -> None:
        state = _base_state(response_language="FR")
        retriever = _FakeRetriever()

        with patch(
            "complaints_orchestrator.agents.context_policy_agent.request.urlopen",
            side_effect=OSError("network issue"),
        ):
            with self.assertRaises(RuntimeError):
                run_context_policy(
                    state,
                    signals=ContextPolicySignals(
                        mistral_api_key="test-key",
                        retriever=retriever,
                    ),
                )

    def test_triage_required_before_context_agent_runs(self) -> None:
        state = CaseState.model_validate(
            {
                "input": {
                    "case_id": "CASE-CONTEXT-2",
                    "customer_id": "CUST-1001",
                    "order_id": "ORD-5001",
                    "email_subject": "Need update",
                    "email_body": "Please update me.",
                    "channel": "EMAIL",
                    "received_at": "2026-02-19T13:30:00Z",
                }
            }
        )

        with self.assertRaises(ValueError):
            run_context_policy(
                state,
                signals=ContextPolicySignals(mistral_api_key="test-key", retriever=_FakeRetriever()),
            )


if __name__ == "__main__":
    unittest.main()

