"""Case history mock read tools."""

from __future__ import annotations

from complaints_orchestrator.tools.data_store import load_json_records


def get_case_history(customer_id: str) -> dict[str, object]:
    cases = load_json_records("mock_cases.json")
    customer_cases = [row for row in cases if row.get("customer_id") == customer_id]
    output_cases = []
    open_case_count = 0
    recent_escalations_count = 0

    for row in customer_cases:
        if row.get("status") == "OPEN":
            open_case_count += 1
        if row.get("status") == "ESCALATED":
            recent_escalations_count += 1
        output_cases.append(
            {
                "case_id": row["case_id"],
                "order_id": row["order_id"],
                "complaint_type": row["complaint_type"],
                "decision": row["decision"],
                "status": row["status"],
                "opened_at": row["opened_at"],
            }
        )
    return {
        "customer_id": customer_id,
        "open_case_count": open_case_count,
        "recent_escalations_count": recent_escalations_count,
        "cases": output_cases,
    }
