"""Domain enums and constants for the complaints orchestrator."""

from __future__ import annotations

from enum import Enum


class ResponseLanguage(str, Enum):
    FR = "FR"
    EN = "EN"


class DecisionType(str, Enum):
    REFUND = "REFUND"
    VOUCHER = "VOUCHER"
    EXCHANGE = "EXCHANGE"
    ESCALATE = "ESCALATE"
    INFO_ONLY = "INFO_ONLY"


class RouteType(str, Enum):
    ESCALATE_IMMEDIATE = "ESCALATE_IMMEDIATE"
    NEED_CONTEXT = "NEED_CONTEXT"
    HITL_REVIEW = "HITL_REVIEW"
    READY_TO_FINALIZE = "READY_TO_FINALIZE"


class UrgencyLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CaseStatus(str, Enum):
    NEW = "NEW"
    IN_PROGRESS = "IN_PROGRESS"
    PENDING_HITL = "PENDING_HITL"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"
    CLOSED = "CLOSED"


class SentimentLabel(str, Enum):
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"


class RiskFlag(str, Enum):
    LEGAL_THREAT = "LEGAL_THREAT"
    PUBLIC_EXPOSURE = "PUBLIC_EXPOSURE"
    REPEAT_CLAIM = "REPEAT_CLAIM"
    HIGH_AMOUNT_RISK = "HIGH_AMOUNT_RISK"

