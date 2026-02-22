"""tests for persistent memory behavior."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from complaints_orchestrator.memory.seed_memory import seed  
from complaints_orchestrator.memory.store import MemoryStore  

class TestMemoryStore(unittest.TestCase):
    def test_preferred_language_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "memory.db")
            store = MemoryStore(db_path=db_path)
            store.upsert_customer_memory("CUST-X", "FR", 0.0)
            self.assertEqual(store.get_preferred_language("CUST-X"), "FR")

    def test_ninety_day_total_filters_old_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "memory.db")
            store = MemoryStore(db_path=db_path)

            recent = (datetime.now(UTC) - timedelta(days=10)).isoformat()
            old = (datetime.now(UTC) - timedelta(days=140)).isoformat()

            store.upsert_case_memory(
                case_id="CASE-RECENT",
                customer_id="CUST-X",
                decision="VOUCHER",
                status="RESOLVED",
                compensation_value=30.0,
                opened_at=recent,
            )
            store.upsert_case_memory(
                case_id="CASE-OLD",
                customer_id="CUST-X",
                decision="VOUCHER",
                status="RESOLVED",
                compensation_value=200.0,
                opened_at=old,
            )

            self.assertEqual(store.get_ninety_day_compensation_total("CUST-X"), 30.0)

    def test_raw_email_persistence_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "memory.db")
            store = MemoryStore(db_path=db_path)

            with self.assertRaises(ValueError):
                store.upsert_case_memory(
                    case_id="CASE-SEC",
                    customer_id="CUST-X",
                    decision="ESCALATE",
                    status="ESCALATED",
                    compensation_value=0.0,
                    opened_at=datetime.now(UTC).isoformat(),
                    summary_payload={"email_body": "forbidden"},
                )

    def test_seed_script_populates_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "seeded.db")
            seed(db_path=db_path)
            store = MemoryStore(db_path=db_path)

            preferred = store.get_preferred_language("CUST-1001")
            total = store.get_ninety_day_compensation_total("CUST-1001")

            self.assertEqual(preferred, "FR")
            self.assertGreaterEqual(total, 0.0)

    def test_finalize_update_writes_case_and_customer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "finalize.db")
            store = MemoryStore(db_path=db_path)

            recent = (datetime.now(UTC) - timedelta(days=1)).isoformat()
            store.record_finalize_update(
                case_id="CASE-FINAL-1",
                customer_id="CUST-FINAL",
                decision="VOUCHER",
                status="RESOLVED",
                compensation_value=25.0,
                opened_at=recent,
                preferred_language="EN",
                summary_payload={"summary": "no raw email stored"},
            )

            self.assertEqual(store.get_preferred_language("CUST-FINAL"), "EN")
            self.assertEqual(store.get_ninety_day_compensation_total("CUST-FINAL"), 25.0)


if __name__ == "__main__":
    unittest.main()
