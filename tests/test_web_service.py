"""Tests for web service adapters."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from complaints_orchestrator.config import AppConfig  # noqa: E402
from complaints_orchestrator.graph import GraphDependencies  # noqa: E402
from complaints_orchestrator.state import CaseState  # noqa: E402
from complaints_orchestrator.web.schemas import RunCaseRequest  # noqa: E402
from complaints_orchestrator.web.service import (  # noqa: E402
    WebRuntime,
    load_scenario_previews,
    run_case,
)


def _runtime() -> WebRuntime:
    config = AppConfig(
        llm_provider="mistral",
        mistral_api_key="test-key",
        model_name="mistral-small-latest",
        embedding_model="mistral-embed",
        chroma_dir="./storage/chroma",
        sqlite_path="./storage/complaints_memory.db",
        hitl_amount_threshold=150.0,
        low_confidence_threshold=0.55,
        log_level="INFO",
    )
    return WebRuntime(config=config, deps=GraphDependencies())


def _graph_final_state() -> CaseState:
    return CaseState.model_validate(
        {
            "input": {
                "case_id": "WEB_CASE_1",
                "customer_id": "CUST-1001",
                "order_id": "ORD-5001",
                "email_subject": "Refund request",
                "email_body": "My item is defective.",
                "channel": "EMAIL",
                "received_at": "2026-02-22T11:00:00Z",
            },
            "triage": {
                "complaint_type": "DEFECTIVE_ITEM",
                "sentiment": "NEGATIVE",
                "urgency": "HIGH",
                "detected_language": "EN",
                "response_language": "EN",
                "risk_flags": [],
                "triage_plan": "Retrieve context and resolve.",
                "route_decision": "NEED_CONTEXT",
                "triage_confidence": 0.91,
            },
            "context": {
                "customer_context": {
                    "customer_id": "CUST-1001",
                    "preferred_language": "EN",
                    "loyalty_tier": "GOLD",
                    "account_age_days": 240,
                    "lifetime_orders": 10,
                    "ninety_day_compensation_total": 0.0,
                    "fraud_watch": False,
                },
                "order_context": {
                    "order_id": "ORD-5001",
                    "currency": "EUR",
                    "order_total": 90.0,
                    "item_count": 1,
                    "status": "DELIVERED",
                },
                "case_history_summary": {
                    "customer_id": "CUST-1001",
                    "total_cases": 1,
                    "open_case_count": 0,
                    "recent_escalations_count": 0,
                    "latest_case_decision": "INFO_ONLY",
                    "latest_case_status": "RESOLVED",
                    "repeat_claim_suspected": False,
                },
                "policy_constraints": ["Refund is allowed for confirmed defects."],
                "policy_source_ids": ["REFUND_POLICY_EN"],
                "rag_snippets": ["Refund allowed when defect confirmed."],
                "context_confidence": 0.88,
            },
            "resolution": {
                "decision": "REFUND",
                "rationale": "Refund selected based on policy and order status.",
                "hitl_required": False,
                "hitl_reason": None,
                "tool_actions": [
                    {
                        "tool_name": "issue_refund",
                        "status": "ISSUED",
                        "reference_id": "RFD-5001",
                        "confirmation_message": "Refund issued for order ORD-5001 (90.0 EUR).",
                        "action_value": 90.0,
                        "action_currency": "EUR",
                    }
                ],
                "response_subject": "Refund confirmation",
                "response_body": "We have issued your refund.",
                "resolution_confidence": 0.9,
            },
            "finalize": {
                "status": "RESOLVED",
                "memory_updates": {
                    "decision": "REFUND",
                    "status": "RESOLVED",
                    "compensation_value": 90.0,
                    "preferred_language": "EN",
                    "output_guard_passed": True,
                },
                "case_summary": "Case WEB_CASE_1: DEFECTIVE_ITEM -> REFUND (RESOLVED)",
            },
            "security_events": ["INGEST_STARTED", "FINALIZE_COMPLETED"],
            "output_guard_passed": True,
        }
    )


class TestWebService(unittest.TestCase):
    def test_run_case_maps_graph_state_to_response_contract(self) -> None:
        request_payload = RunCaseRequest(
            case_id="web-case-raw",
            customer_id="CUST-1001",
            order_id="ORD-5001",
            email_subject="Refund request",
            email_body="My item is defective.",
            channel="EMAIL",
        )
        runtime = _runtime()
        captured: dict[str, object] = {}

        def _fake_run_graph(state: CaseState, deps=None) -> CaseState:
            captured["state"] = state
            captured["deps"] = deps
            return _graph_final_state()

        with patch("complaints_orchestrator.web.service.run_graph", side_effect=_fake_run_graph):
            response = run_case(request_payload=request_payload, runtime=runtime)

        self.assertEqual(response.case_id, "WEB_CASE_1")
        self.assertEqual(response.resolution["decision"], "REFUND")
        self.assertIsInstance(response.runtime_ms, int)
        self.assertGreaterEqual(response.runtime_ms, 0)
        self.assertIs(captured["deps"], runtime.deps)

        captured_state = captured["state"]
        self.assertIsInstance(captured_state, CaseState)
        assert isinstance(captured_state, CaseState)
        self.assertEqual(captured_state.input.case_id, "WEB_CASE_RAW")
        self.assertEqual(captured_state.input.customer_id, "CUST-1001")

    def test_load_scenario_previews_parses_expected_fields(self) -> None:
        payload = [
            {
                "id": "late_delivery",
                "title": "Late delivery",
                "email_subject": "Where is my order",
                "email_body": "Need an update",
                "customer_id": "CUST-2001",
                "order_id": "ORD-9001",
                "preferred_language": "EN",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "scenarios.json"
            file_path.write_text(json.dumps(payload), encoding="utf-8")

            previews = load_scenario_previews(scenarios_file=file_path)

        self.assertEqual(len(previews), 1)
        self.assertEqual(previews[0].id, "late_delivery")
        self.assertEqual(previews[0].customer_id, "CUST-2001")

    def test_load_scenario_previews_rejects_non_list_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "scenarios.json"
            file_path.write_text('{"id":"invalid"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_scenario_previews(scenarios_file=file_path)

    def test_load_scenario_previews_supports_eval_format(self) -> None:
        payload = [
            {
                "id": "eval_case_1",
                "title": "Eval scenario",
                "input": {
                    "customer_id": "CUST-3001",
                    "order_id": "ORD-7001",
                    "email_subject": "Need a refund",
                    "email_body": "My item was damaged.",
                    "channel": "EMAIL",
                },
                "expected": {
                    "route_decision": "NEED_CONTEXT",
                    "response_language": "FR",
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "eval_scenarios.json"
            file_path.write_text(json.dumps(payload), encoding="utf-8")
            previews = load_scenario_previews(scenarios_file=file_path)

        self.assertEqual(len(previews), 1)
        self.assertEqual(previews[0].id, "eval_case_1")
        self.assertEqual(previews[0].customer_id, "CUST-3001")
        self.assertEqual(previews[0].email_subject, "Need a refund")
        self.assertEqual(previews[0].preferred_language, "FR")
