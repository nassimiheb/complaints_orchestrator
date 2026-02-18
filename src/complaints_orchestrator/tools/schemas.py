"""Pydantic schemas for mock tool inputs and outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetCustomerProfileInput(ToolSchema):
    customer_id: str = Field(min_length=1)


class GetCustomerProfileOutput(ToolSchema):
    customer_id: str
    preferred_language: str
    loyalty_tier: str
    account_age_days: int = Field(ge=0)
    lifetime_orders: int = Field(ge=0)
    ninety_day_compensation_total: float = Field(ge=0.0)
    fraud_watch: bool


class GetOrderDetailsInput(ToolSchema):
    order_id: str = Field(min_length=1)


class GetOrderDetailsOutput(ToolSchema):
    order_id: str
    customer_id: str
    currency: str
    order_total: float = Field(ge=0.0)
    item_count: int = Field(ge=0)
    status: str


class CaseHistoryRecord(ToolSchema):
    case_id: str
    order_id: str
    complaint_type: str
    decision: str
    status: str
    opened_at: str


class GetCaseHistoryInput(ToolSchema):
    customer_id: str = Field(min_length=1)


class GetCaseHistoryOutput(ToolSchema):
    customer_id: str
    open_case_count: int = Field(ge=0)
    recent_escalations: int = Field(ge=0)
    cases: list[CaseHistoryRecord]


class CreateCompensationInput(ToolSchema):
    case_id: str = Field(min_length=1)
    type: str
    value: float = Field(ge=0.0)
    currency: str = Field(default="EUR")


class CreateCompensationOutput(ToolSchema):
    compensation_id: str
    status: str
    applied_value: float = Field(ge=0.0)
    currency: str
    created_at: str


class IssueRefundInput(ToolSchema):
    order_id: str = Field(min_length=1)
    amount: float = Field(ge=0.0)
    currency: str = Field(default="EUR")


class IssueRefundOutput(ToolSchema):
    refund_id: str
    status: str
    amount: float = Field(ge=0.0)
    currency: str
    processed_at: str


class CreateSupportTicketInput(ToolSchema):
    case_payload: dict[str, Any]
    priority: str


class CreateSupportTicketOutput(ToolSchema):
    ticket_id: str
    status: str
    queue: str
    created_at: str

