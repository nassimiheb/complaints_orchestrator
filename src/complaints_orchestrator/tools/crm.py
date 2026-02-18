"""CRM mock read tools."""

from __future__ import annotations

from complaints_orchestrator.tools.data_store import load_json_records


def get_customer_profile(customer_id: str) -> dict[str, object]:
    customers = load_json_records("mock_customers.json")
    for customer in customers:
        if customer.get("customer_id") == customer_id:
            return {
                "customer_id": customer["customer_id"],
                "preferred_language": customer["preferred_language"],
                "loyalty_tier": customer["loyalty_tier"],
                "account_age_days": customer["account_age_days"],
                "lifetime_orders": customer["lifetime_orders"],
                "ninety_day_compensation_total": customer["ninety_day_compensation_total"],
                "fraud_watch": customer["fraud_watch"],
            }
    raise LookupError(f"Customer not found: {customer_id}")

