"""OMS mock read tools."""

from __future__ import annotations

from complaints_orchestrator.tools.data_store import load_json_records


def get_order_details(order_id: str) -> dict[str, object]:
    orders = load_json_records("mock_orders.json")
    for order in orders:
        if order.get("order_id") == order_id:
            return {
                "order_id": order["order_id"],
                "customer_id": order["customer_id"],
                "currency": order["currency"],
                "order_total": order["order_total"],
                "item_count": order["item_count"],
                "status": order["status"],
            }
    raise LookupError(f"Order not found: {order_id}")

