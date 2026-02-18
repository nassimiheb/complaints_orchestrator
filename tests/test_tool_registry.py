"""Phase 2 tests for tool registry permissions and validation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from complaints_orchestrator.tools.registry import (  
    ToolPermissionError,
    call_tool,
    list_tools_for_role,
)


class TestToolRegistry(unittest.TestCase):
    def test_read_tools_available_for_context_role(self) -> None:
        tools = list_tools_for_role("context_policy_node")
        self.assertIn("get_customer_profile", tools)
        self.assertIn("get_order_details", tools)
        self.assertIn("get_case_history", tools)

    def test_action_tool_blocked_outside_resolution_node(self) -> None:
        with self.assertRaises(ToolPermissionError):
            call_tool(
                tool_name="issue_refund",
                role="context_policy_node",
                payload={"order_id": "ORD-5001", "amount": 12.0, "currency": "EUR"},
            )

    def test_tool_arguments_are_schema_validated(self) -> None:
        with self.assertRaises(ValidationError):
            call_tool(
                tool_name="get_order_details",
                role="context_policy_node",
                payload={"order_id": ""},
            )

    def test_read_tool_success(self) -> None:
        output = call_tool(
            tool_name="get_customer_profile",
            role="context_policy_node",
            payload={"customer_id": "CUST-1001"},
        )
        self.assertEqual(output["customer_id"], "CUST-1001")
        self.assertIn("preferred_language", output)

    def test_action_tool_success_from_resolution_node(self) -> None:
        output = call_tool(
            tool_name="create_support_ticket",
            role="resolution_node",
            payload={"case_payload": {"case_id": "CASE-9001"}, "priority": "HIGH"},
        )
        self.assertEqual(output["status"], "OPEN")
        self.assertEqual(output["queue"], "LEGAL")


if __name__ == "__main__":
    unittest.main()
