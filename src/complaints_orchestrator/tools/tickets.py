"""Support ticket mock action tool."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5


def create_support_ticket(case_payload: dict[str, object], priority: str) -> dict[str, str]:
    case_id = str(case_payload.get("case_id", "UNKNOWN"))
    seed = f"{case_id}:{priority}"
    ticket_id = f"TCK-{str(uuid5(NAMESPACE_URL, seed)).split('-')[0].upper()}"

    queue = "LEGAL" if priority.upper() in {"HIGH", "CRITICAL"} else "STANDARD"
    return {
        "ticket_id": ticket_id,
        "status": "OPEN",
        "queue": queue,
        "created_at": datetime.now(UTC).isoformat(),
    }

