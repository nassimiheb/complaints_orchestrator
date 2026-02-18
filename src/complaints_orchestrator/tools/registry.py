"""Tool registry with role permissions and schema validation wrappers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from complaints_orchestrator.tools.actions import create_compensation, issue_refund
from complaints_orchestrator.tools.cases import get_case_history
from complaints_orchestrator.tools.crm import get_customer_profile
from complaints_orchestrator.tools.oms import get_order_details
from complaints_orchestrator.tools.schemas import (
    CreateCompensationInput,
    CreateCompensationOutput,
    CreateSupportTicketInput,
    CreateSupportTicketOutput,
    GetCaseHistoryInput,
    GetCaseHistoryOutput,
    GetCustomerProfileInput,
    GetCustomerProfileOutput,
    GetOrderDetailsInput,
    GetOrderDetailsOutput,
    IssueRefundInput,
    IssueRefundOutput,
)
from complaints_orchestrator.tools.tickets import create_support_ticket
from complaints_orchestrator.utils.retry import retry

LOGGER = logging.getLogger(__name__)


class ToolPermissionError(PermissionError):
    """Raised when a node role calls a forbidden tool."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    allowed_roles: frozenset[str]
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Any


def _wrap_customer_profile(args: BaseModel) -> dict[str, object]:
    return get_customer_profile(customer_id=args.customer_id)


def _wrap_order_details(args: BaseModel) -> dict[str, object]:
    return get_order_details(order_id=args.order_id)


def _wrap_case_history(args: BaseModel) -> dict[str, object]:
    return get_case_history(customer_id=args.customer_id)


def _wrap_create_compensation(args: BaseModel) -> dict[str, object]:
    return create_compensation(case_id=args.case_id, type=args.type, value=args.value, currency=args.currency)


def _wrap_issue_refund(args: BaseModel) -> dict[str, object]:
    return issue_refund(order_id=args.order_id, amount=args.amount, currency=args.currency)


def _wrap_create_ticket(args: BaseModel) -> dict[str, object]:
    return create_support_ticket(case_payload=args.case_payload, priority=args.priority)


TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "get_customer_profile": ToolDefinition(
        name="get_customer_profile",
        allowed_roles=frozenset({"context_policy_node"}),
        input_model=GetCustomerProfileInput,
        output_model=GetCustomerProfileOutput,
        handler=_wrap_customer_profile,
    ),
    "get_order_details": ToolDefinition(
        name="get_order_details",
        allowed_roles=frozenset({"context_policy_node"}),
        input_model=GetOrderDetailsInput,
        output_model=GetOrderDetailsOutput,
        handler=_wrap_order_details,
    ),
    "get_case_history": ToolDefinition(
        name="get_case_history",
        allowed_roles=frozenset({"context_policy_node"}),
        input_model=GetCaseHistoryInput,
        output_model=GetCaseHistoryOutput,
        handler=_wrap_case_history,
    ),
    "create_compensation": ToolDefinition(
        name="create_compensation",
        allowed_roles=frozenset({"resolution_node"}),
        input_model=CreateCompensationInput,
        output_model=CreateCompensationOutput,
        handler=_wrap_create_compensation,
    ),
    "issue_refund": ToolDefinition(
        name="issue_refund",
        allowed_roles=frozenset({"resolution_node"}),
        input_model=IssueRefundInput,
        output_model=IssueRefundOutput,
        handler=_wrap_issue_refund,
    ),
    "create_support_ticket": ToolDefinition(
        name="create_support_ticket",
        allowed_roles=frozenset({"resolution_node"}),
        input_model=CreateSupportTicketInput,
        output_model=CreateSupportTicketOutput,
        handler=_wrap_create_ticket,
    ),
}


def list_tools_for_role(role: str) -> list[str]:
    return sorted([name for name, spec in TOOL_REGISTRY.items() if role in spec.allowed_roles])


def call_tool(tool_name: str, role: str, payload: dict[str, Any]) -> dict[str, Any]:
    if tool_name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool: {tool_name}")

    tool = TOOL_REGISTRY[tool_name]
    if role not in tool.allowed_roles:
        raise ToolPermissionError(f"Role '{role}' cannot call tool '{tool_name}'")

    validated_input = tool.input_model.model_validate(payload)

    def _execute() -> dict[str, Any]:
        raw_output = tool.handler(validated_input)
        validated_output = tool.output_model.model_validate(raw_output)
        return validated_output.model_dump()

    output = retry(_execute, retries=3, base_delay_seconds=0.05)
    LOGGER.info(
        "Tool call succeeded",
        extra={"tool_name": tool_name, "role": role},
    )
    return output

