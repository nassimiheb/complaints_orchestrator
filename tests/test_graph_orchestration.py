""" tests for LangGraph orchestration and finalize behavior."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from complaints_orchestrator.constants import CaseStatus  # noqa: E402
from complaints_orchestrator.graph import (  # noqa: E402
    GraphDependencies,
    finalize_node,
    run_graph,
)
from complaints_orchestrator.memory.store import MemoryStore  # noqa: E402
from complaints_orchestrator.state import (  # noqa: E402
    CaseState,
    ContextOutput,
    ResolutionOutput,
    ToolActionRecord,
    TriageOutput,
)


def _base_state() -> CaseState:
    return CaseState.model_validate(
        {
            "input": {
                "case_id": "CASE-GRAPH-1",
                "customer_id": "CUST-1001",
                "order_id": "ORD-5001",
                "email_subject": "Need support",
                "email_body": "My item is defective and I need help.",
                "channel": "EMAIL",
                "received_at": "2026-02-20T10:00:00Z",
            }
        }
    )


class TestGraphOrchestration(unittest.TestCase):
    def test_escalate_immediate_route_skips_context_node(self) -> None:
        state = _base_state()
        calls: list[str] = []

        def _fake_triage(run_state: CaseState, signals=None) -> CaseState:
            calls.append("triage")
            run_state.triage = TriageOutput.model_validate(
                {
                    "complaint_type": "PUBLIC_COMPLAINT",
                    "sentiment": "NEGATIVE",
                    "urgency": "CRITICAL",
                    "detected_language": "EN",
                    "response_language": "EN",
                    "risk_flags": ["LEGAL_THREAT"],
                    "triage_plan": "Escalate immediately.",
                    "route_decision": "ESCALATE_IMMEDIATE",
                    "triage_confidence": 0.9,
                }
            )
            return run_state

        def _fake_resolution(run_state: CaseState, signals=None) -> CaseState:
            calls.append("resolution")
            self.assertIsNotNone(run_state.context)
            run_state.output_guard_passed = True
            run_state.resolution = ResolutionOutput.model_validate(
                {
                    "decision": "ESCALATE",
                    "rationale": "Immediate escalation due to legal/public exposure risk.",
                    "hitl_required": True,
                    "hitl_reason": "LEGAL_OR_PUBLIC_RISK",
                    "tool_actions": [
                        {
                            "tool_name": "create_support_ticket",
                            "status": "OPEN",
                            "reference_id": "TCK-100",
                            "confirmation_message": "Support ticket opened in queue LEGAL.",
                        }
                    ],
                    "response_subject": "Update on your complaint",
                    "response_body": "Your case has been escalated to a specialist.",
                    "resolution_confidence": 0.89,
                }
            )
            return run_state

        with patch("complaints_orchestrator.graph.run_triage", side_effect=_fake_triage):
            with patch(
                "complaints_orchestrator.graph.run_context_policy",
                side_effect=AssertionError("context_policy_node should not run for ESCALATE_IMMEDIATE"),
            ):
                with patch("complaints_orchestrator.graph.run_resolution", side_effect=_fake_resolution):
                    result = run_graph(state, deps=GraphDependencies())

        self.assertEqual(calls, ["triage", "resolution"])
        self.assertIn("GRAPH_ROUTE_ESCALATE_IMMEDIATE", result.security_events)
        self.assertIn("GRAPH_ESCALATE_IMMEDIATE_CONTEXT_STUBBED", result.security_events)
        self.assertIsNotNone(result.finalize)
        assert result.finalize is not None
        self.assertEqual(result.finalize.status, CaseStatus.ESCALATED)

    def test_normal_route_runs_context_before_resolution(self) -> None:
        state = _base_state()
        calls: list[str] = []

        def _fake_triage(run_state: CaseState, signals=None) -> CaseState:
            calls.append("triage")
            run_state.triage = TriageOutput.model_validate(
                {
                    "complaint_type": "DEFECTIVE_ITEM",
                    "sentiment": "NEGATIVE",
                    "urgency": "HIGH",
                    "detected_language": "EN",
                    "response_language": "EN",
                    "risk_flags": [],
                    "triage_plan": "Fetch context and policy constraints.",
                    "route_decision": "NEED_CONTEXT",
                    "triage_confidence": 0.9,
                }
            )
            return run_state

        def _fake_context(run_state: CaseState, signals=None) -> CaseState:
            calls.append("context")
            run_state.context = ContextOutput.model_validate(
                {
                    "customer_context": {
                        "customer_id": "CUST-1001",
                        "preferred_language": "EN",
                        "loyalty_tier": "GOLD",
                        "account_age_days": 300,
                        "lifetime_orders": 12,
                        "ninety_day_compensation_total": 5.0,
                        "fraud_watch": False,
                    },
                    "order_context": {
                        "order_id": "ORD-5001",
                        "currency": "EUR",
                        "order_total": 80.0,
                        "item_count": 1,
                        "status": "DELIVERED",
                    },
                    "case_history_summary": {
                        "customer_id": "CUST-1001",
                        "total_cases": 1,
                        "open_case_count": 0,
                        "recent_escalations_count": 0,
                        "latest_case_decision": "VOUCHER",
                        "latest_case_status": "CLOSED",
                        "repeat_claim_suspected": False,
                    },
                    "policy_constraints": ["Refund is allowed for confirmed defects."],
                    "policy_source_ids": ["REFUND_POLICY_EN"],
                    "rag_snippets": ["Refund allowed when defect confirmed."],
                    "context_confidence": 0.87,
                }
            )
            return run_state

        def _fake_resolution(run_state: CaseState, signals=None) -> CaseState:
            calls.append("resolution")
            self.assertIsNotNone(run_state.context)
            run_state.output_guard_passed = True
            run_state.resolution = ResolutionOutput.model_validate(
                {
                    "decision": "REFUND",
                    "rationale": "Refund selected from context and policy.",
                    "hitl_required": False,
                    "hitl_reason": None,
                    "tool_actions": [
                        {
                            "tool_name": "issue_refund",
                            "status": "ISSUED",
                            "reference_id": "RFD-100",
                            "confirmation_message": "Refund issued for order ORD-5001 (80.0 EUR).",
                        }
                    ],
                    "response_subject": "Refund confirmation",
                    "response_body": "We have issued your refund.",
                    "resolution_confidence": 0.9,
                }
            )
            return run_state

        with patch("complaints_orchestrator.graph.run_triage", side_effect=_fake_triage):
            with patch("complaints_orchestrator.graph.run_context_policy", side_effect=_fake_context):
                with patch("complaints_orchestrator.graph.run_resolution", side_effect=_fake_resolution):
                    result = run_graph(state, deps=GraphDependencies())

        self.assertEqual(calls, ["triage", "context", "resolution"])
        self.assertIn("GRAPH_ROUTE_NEED_CONTEXT", result.security_events)
        self.assertIsNotNone(result.finalize)
        assert result.finalize is not None
        self.assertEqual(result.finalize.status, CaseStatus.RESOLVED)

    def test_finalize_persists_structured_summary_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "test.db")
            store = MemoryStore(db_path=db_path)
            state = _base_state()
            state.triage = TriageOutput.model_validate(
                {
                    "complaint_type": "LATE_DELIVERY",
                    "sentiment": "NEGATIVE",
                    "urgency": "MEDIUM",
                    "detected_language": "EN",
                    "response_language": "EN",
                    "risk_flags": [],
                    "triage_plan": "Proceed with compensation path.",
                    "route_decision": "NEED_CONTEXT",
                    "triage_confidence": 0.88,
                }
            )
            state.context = ContextOutput.model_validate(
                {
                    "customer_context": {
                        "customer_id": "CUST-1001",
                        "preferred_language": "EN",
                        "loyalty_tier": "STANDARD",
                        "account_age_days": 120,
                        "lifetime_orders": 4,
                        "ninety_day_compensation_total": 0.0,
                        "fraud_watch": False,
                    },
                    "order_context": {
                        "order_id": "ORD-5001",
                        "currency": "EUR",
                        "order_total": 60.0,
                        "item_count": 1,
                        "status": "DELIVERED",
                    },
                    "case_history_summary": {
                        "customer_id": "CUST-1001",
                        "total_cases": 1,
                        "open_case_count": 0,
                        "recent_escalations_count": 0,
                        "latest_case_decision": "INFO_ONLY",
                        "latest_case_status": "CLOSED",
                        "repeat_claim_suspected": False,
                    },
                    "policy_constraints": ["Compensation is acceptable for delivery delay."],
                    "policy_source_ids": ["COMPENSATION_POLICY_EN"],
                    "rag_snippets": ["Compensation can be offered for delays."],
                    "context_confidence": 0.82,
                }
            )
            state.resolution = ResolutionOutput.model_validate(
                {
                    "decision": "VOUCHER",
                    "rationale": "Voucher selected for delay inconvenience.",
                    "hitl_required": False,
                    "hitl_reason": None,
                    "tool_actions": [
                        ToolActionRecord(
                            tool_name="create_compensation",
                            status="CREATED",
                            reference_id="CMP-200",
                            confirmation_message="Voucher created for 25.0 EUR.",
                        ).model_dump()
                    ],
                    "response_subject": "Compensation update",
                    "response_body": "We created a voucher for your next order.",
                    "resolution_confidence": 0.86,
                }
            )
            state.output_guard_passed = True

            captured: dict[str, object] = {}
            original_method = store.record_finalize_update

            def _capture(*args, **kwargs):
                captured["summary_payload"] = kwargs.get("summary_payload")
                return original_method(*args, **kwargs)

            with patch.object(store, "record_finalize_update", side_effect=_capture):
                finalize_node(state, deps=GraphDependencies(memory_store=store))

            self.assertIn("summary_payload", captured)
            assert isinstance(captured["summary_payload"], dict)
            summary_payload = captured["summary_payload"]
            assert isinstance(summary_payload, dict)
            self.assertNotIn("email_body", summary_payload)
            self.assertNotIn("raw_email", summary_payload)

            self.assertEqual(store.get_preferred_language("CUST-1001"), "EN")
            self.assertAlmostEqual(store.get_ninety_day_compensation_total("CUST-1001"), 25.0, places=2)
            self.assertIsNotNone(state.finalize)
            assert state.finalize is not None
            self.assertEqual(state.finalize.status, CaseStatus.RESOLVED)
            self.assertIn("FINALIZE_MEMORY_UPDATED", state.security_events)


if __name__ == "__main__":
    unittest.main()
