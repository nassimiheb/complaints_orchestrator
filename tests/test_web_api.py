"""Tests for web API endpoints."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from complaints_orchestrator.config import AppConfig  # noqa: E402
from complaints_orchestrator.graph import GraphDependencies  # noqa: E402
from complaints_orchestrator.web.app import create_app  # noqa: E402
from complaints_orchestrator.web.schemas import RunCaseResponse  # noqa: E402
from complaints_orchestrator.web.service import WebRuntime  # noqa: E402


def _runtime() -> WebRuntime:
    config = AppConfig(
        llm_provider="mistral",
        mistral_api_key="test-key",
        model_name="mistral-small-latest",
        embedding_model="mistral-embed",
        chroma_dir="./storage/chroma",
        sqlite_path="./storage/complaints_memory.db",
        hitl_amount_threshold=150.0,
        low_confidence_threshold=0.55,
        log_level="INFO",
    )
    return WebRuntime(config=config, deps=GraphDependencies())


class TestWebAPI(unittest.TestCase):
    def test_health_endpoint_returns_ok(self) -> None:
        app = create_app(runtime=_runtime(), ensure_index_if_missing=False)
        with TestClient(app) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_scenarios_endpoint_returns_payload(self) -> None:
        scenarios = [
            {
                "id": "scenario_1",
                "title": "Sample",
                "customer_id": "CUST-1001",
                "order_id": "ORD-5001",
                "email_subject": "Need help",
                "email_body": "My order is late.",
                "preferred_language": "EN",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "scenarios.json"
            file_path.write_text(json.dumps(scenarios), encoding="utf-8")
            app = create_app(
                runtime=_runtime(),
                scenarios_file=file_path,
                ensure_index_if_missing=False,
            )
            with TestClient(app) as client:
                response = client.get("/api/scenarios")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["id"], "scenario_1")

    def test_run_case_endpoint_returns_response_model(self) -> None:
        app = create_app(runtime=_runtime(), ensure_index_if_missing=False)
        mocked_response = RunCaseResponse(
            case_id="WEB_CASE_1",
            triage={"complaint_type": "LATE_DELIVERY"},
            context={},
            resolution={"decision": "INFO_ONLY"},
            finalize={"status": "RESOLVED"},
            security_events=["FINALIZE_COMPLETED"],
            output_guard_passed=True,
            runtime_ms=42,
        )
        payload = {
            "customer_id": "CUST-1002",
            "order_id": "ORD-5002",
            "email_subject": "Order delay",
            "email_body": "Where is my order?",
            "channel": "EMAIL",
        }

        with patch("complaints_orchestrator.web.app.run_case", return_value=mocked_response) as run_case_mock:
            with TestClient(app) as client:
                response = client.post("/api/cases/run", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["case_id"], "WEB_CASE_1")
        run_case_mock.assert_called_once()

    def test_run_case_endpoint_validation_error(self) -> None:
        app = create_app(runtime=_runtime(), ensure_index_if_missing=False)
        payload = {
            "customer_id": "CUST-1002",
            "order_id": "ORD-5002",
            "email_subject": "Order delay",
            "channel": "EMAIL",
        }

        with TestClient(app) as client:
            response = client.post("/api/cases/run", json=payload)

        self.assertEqual(response.status_code, 422)
