"""Graph state contract for the complaints orchestrator."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from complaints_orchestrator.constants import (
    CaseStatus,
    DecisionType,
    ResponseLanguage,
    RiskFlag,
    RouteType,
    SentimentLabel,
    UrgencyLevel,
)


class StateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CaseInput(StateModel):
    case_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    email_subject: str = Field(min_length=1)
    email_body: str = Field(min_length=1)
    channel: str
    received_at: str


class TriageOutput(StateModel):
    complaint_type: str
    sentiment: SentimentLabel
    urgency: UrgencyLevel
    detected_language: ResponseLanguage
    response_language: ResponseLanguage
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    triage_plan: str
    route_decision: RouteType
    triage_confidence: float = Field(ge=0.0, le=1.0)


class ContextOutput(StateModel):
    customer_context: dict[str, str | int | float | bool]
    order_context: dict[str, str | int | float | bool]
    case_history_summary: dict[str, str | int | float | bool]
    policy_constraints: list[str] = Field(default_factory=list)
    policy_source_ids: list[str] = Field(default_factory=list)
    rag_snippets: list[str] = Field(default_factory=list)
    context_confidence: float = Field(ge=0.0, le=1.0)


class ToolActionRecord(StateModel):
    tool_name: str
    status: str
    reference_id: str
    confirmation_message: str
    action_value: float | None = None
    action_currency: str | None = None


class ResolutionOutput(StateModel):
    decision: DecisionType
    rationale: str
    hitl_required: bool
    hitl_reason: str | None = None
    tool_actions: list[ToolActionRecord] = Field(default_factory=list)
    response_subject: str
    response_body: str
    resolution_confidence: float = Field(ge=0.0, le=1.0)


class FinalizeOutput(StateModel):
    status: CaseStatus
    memory_updates: dict[str, str | int | float | bool]
    case_summary: str


class CaseState(StateModel):
    input: CaseInput
    triage: TriageOutput | None = None
    context: ContextOutput | None = None
    resolution: ResolutionOutput | None = None
    finalize: FinalizeOutput | None = None
    redacted_email_body: str = ""
    security_events: list[str] = Field(default_factory=list)
    output_guard_passed: bool = False
