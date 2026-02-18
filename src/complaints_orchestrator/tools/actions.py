"""Action mock tools for compensation and refunds."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5


def create_compensation(
    case_id: str, type: str, value: float, currency: str = "EUR"
) -> dict[str, object]:
    token = f"{case_id}:{type}:{value:.2f}:{currency}"
    compensation_id = f"CMP-{str(uuid5(NAMESPACE_URL, token)).split('-')[0].upper()}"
    return {
        "compensation_id": compensation_id,
        "status": "CREATED",
        "applied_value": value,
        "currency": currency,
        "created_at": datetime.now(UTC).isoformat(),
    }


def issue_refund(
    order_id: str, amount: float, currency: str = "EUR"
) -> dict[str, object]:
    token = f"{order_id}:{amount:.2f}:{currency}"
    refund_id = f"RFD-{str(uuid5(NAMESPACE_URL, token)).split('-')[0].upper()}"
    return {
        "refund_id": refund_id,
        "status": "ISSUED",
        "amount": amount,
        "currency": currency,
        "processed_at": datetime.now(UTC).isoformat(),
    }
