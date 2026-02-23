"""HTTP request/response schemas for the web UI."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunCaseRequest(BaseModel):
    """Payload accepted by the case execution endpoint."""

    model_config = ConfigDict(extra="forbid")

    case_id: str | None = None
    customer_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    email_subject: str = Field(min_length=1)
    email_body: str = Field(min_length=1)
    channel: str = Field(default="EMAIL", min_length=1)

    @field_validator("case_id", mode="before")
    @classmethod
    def _normalize_case_id(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("customer_id", "order_id", "email_subject", "email_body", "channel", mode="before")
    @classmethod
    def _strip_required(cls, value: object) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Field cannot be empty.")
        return text


class ScenarioPreview(BaseModel):
    """Scenario shape returned to UI for quick prefilling."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    customer_id: str
    order_id: str
    email_subject: str
    email_body: str
    preferred_language: str | None = None


class RunCaseResponse(BaseModel):
    """Structured response returned after graph execution."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    triage: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    resolution: dict[str, Any] | None = None
    finalize: dict[str, Any] | None = None
    security_events: list[str] = Field(default_factory=list)
    output_guard_passed: bool
    runtime_ms: int
