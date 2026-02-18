"""Seed local SQLite memory from mock data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from complaints_orchestrator.memory.store import MemoryStore


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_records(file_name: str) -> list[dict]:
    path = _project_root() / "data" / file_name
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def seed(db_path: str) -> None:
    store = MemoryStore(db_path=db_path)

    customers = _load_records("mock_customers.json")
    for customer in customers:
        store.upsert_customer_memory(
            customer_id=customer["customer_id"],
            preferred_language=customer["preferred_language"],
            ninety_day_compensation_total=float(customer["ninety_day_compensation_total"]),
        )

    cases = _load_records("mock_cases.json")
    for case in cases:
        store.upsert_case_memory(
            case_id=case["case_id"],
            customer_id=case["customer_id"],
            decision=case["decision"],
            status=case["status"],
            compensation_value=float(case["compensation_value"]),
            opened_at=case["opened_at"],
            summary_payload={"source": "seed"},
        )

    # Recompute customer totals based on current cases_memory snapshot.
    for customer in customers:
        total = store.get_ninety_day_compensation_total(customer_id=customer["customer_id"])
        store.upsert_customer_memory(
            customer_id=customer["customer_id"],
            preferred_language=customer["preferred_language"],
            ninety_day_compensation_total=total,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed complaints memory SQLite database.")
    parser.add_argument(
        "--db-path",
        default="./storage/complaints_memory.db",
        help="Path to SQLite memory database.",
    )
    args = parser.parse_args()
    seed(db_path=args.db_path)
    print(f"Memory seeded at {args.db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

