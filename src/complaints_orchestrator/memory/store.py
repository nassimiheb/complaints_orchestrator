"""SQLite persistent memory adapter."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

FORBIDDEN_RAW_EMAIL_KEYS = {"email_body", "raw_email", "raw_email_body"}


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class MemoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        schema_sql = schema_path.read_text(encoding="utf-8")
        with self._connection() as conn:
            conn.executescript(schema_sql)

    @staticmethod
    def _assert_no_raw_email(summary_payload: dict[str, Any] | None) -> None:
        if not summary_payload:
            return
        lowered_keys = {key.lower() for key in summary_payload.keys()}
        if FORBIDDEN_RAW_EMAIL_KEYS.intersection(lowered_keys):
            raise ValueError("Raw email content must never be persisted.")

    def upsert_customer_memory(
        self,
        customer_id: str,
        preferred_language: str,
        ninety_day_compensation_total: float,
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO customers_memory(customer_id, preferred_language, ninety_day_compensation_total, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(customer_id) DO UPDATE SET
                    preferred_language = excluded.preferred_language,
                    ninety_day_compensation_total = excluded.ninety_day_compensation_total,
                    updated_at = excluded.updated_at
                """,
                (customer_id, preferred_language, float(ninety_day_compensation_total), utc_now_iso()),
            )

    def upsert_case_memory(
        self,
        case_id: str,
        customer_id: str,
        decision: str,
        status: str,
        compensation_value: float,
        opened_at: str,
        summary_payload: dict[str, Any] | None = None,
    ) -> None:
        self._assert_no_raw_email(summary_payload)
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO cases_memory(case_id, customer_id, decision, status, compensation_value, opened_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    customer_id = excluded.customer_id,
                    decision = excluded.decision,
                    status = excluded.status,
                    compensation_value = excluded.compensation_value,
                    opened_at = excluded.opened_at,
                    updated_at = excluded.updated_at
                """,
                (
                    case_id,
                    customer_id,
                    decision,
                    status,
                    float(compensation_value),
                    opened_at,
                    utc_now_iso(),
                ),
            )

    def record_finalize_update(
        self,
        case_id: str,
        customer_id: str,
        decision: str,
        status: str,
        compensation_value: float,
        opened_at: str,
        preferred_language: str,
        summary_payload: dict[str, Any] | None = None,
    ) -> None:
        self.upsert_case_memory(
            case_id=case_id,
            customer_id=customer_id,
            decision=decision,
            status=status,
            compensation_value=compensation_value,
            opened_at=opened_at,
            summary_payload=summary_payload,
        )
        total = self.get_ninety_day_compensation_total(customer_id=customer_id)
        self.upsert_customer_memory(
            customer_id=customer_id,
            preferred_language=preferred_language,
            ninety_day_compensation_total=total,
        )

    def get_preferred_language(self, customer_id: str) -> str | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT preferred_language FROM customers_memory WHERE customer_id = ?",
                (customer_id,),
            ).fetchone()
        if row is None:
            return None
        return str(row["preferred_language"])

    def get_ninety_day_compensation_total(self, customer_id: str) -> float:
        cutoff = (datetime.now(UTC) - timedelta(days=90)).isoformat()
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(compensation_value), 0) AS total
                FROM cases_memory
                WHERE customer_id = ? AND opened_at >= ?
                """,
                (customer_id, cutoff),
            ).fetchone()
        if row is None:
            return 0.0
        return float(row["total"])
