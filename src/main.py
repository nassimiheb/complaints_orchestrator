"""CLI entrypoint for the complaints orchestrator."""

from __future__ import annotations

import argparse
import logging

from complaints_orchestrator.config import AppConfig
from complaints_orchestrator.logging_config import configure_logging
from complaints_orchestrator.state import CaseState
from complaints_orchestrator.utils.language import choose_response_language, detect_language
from complaints_orchestrator.utils.output_guard import apply_output_guard
from complaints_orchestrator.utils.pii import redact_for_triage

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="complaints-orchestrator",
        description="Complaints Resolution Orchestrator",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        help="Scenario id to run.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to an optional dotenv file.",
    )
    parser.add_argument(
        "--demo-security",
        action="store_true",
        help="Run a security utility demo.",
    )
    return parser


def _build_demo_state() -> CaseState:
    payload = {
        "input": {
            "case_id": "CASE-DEMO-SEC-1",
            "customer_id": "CUST-1001",
            "order_id": "ORD-5001",
            "email_subject": "Need urgent refund",
            "email_body": (
                "Bonjour, my email is alice.martin@example.com and phone is +33 6 12 34 56 78. "
                "I want a refund for my defective item."
            ),
            "channel": "EMAIL",
            "received_at": "2026-02-19T10:00:00Z",
        },
    }
    return CaseState.model_validate(payload)


def run_security_demo() -> None:
    state = _build_demo_state()

    state.redacted_email_body = redact_for_triage(
        state.input.email_body,
        security_events=state.security_events,
        logger=LOGGER,
    )
    detected_language = detect_language(state.redacted_email_body)
    response_language = choose_response_language(
        detected_language=detected_language,
        preferred_language="EN",
        security_events=state.security_events,
        logger=LOGGER,
    )

    draft_subject = "Complaint update"
    draft_body = (
        "Here is internal score=0.92 and doc_id=REFUND_POLICY_FR.\n"
        '{"refund_id":"RFD-0001","status":"ISSUED"}\n'
        "We are sorry for this issue and we will help you."
    )
    guard = apply_output_guard(
        subject=draft_subject,
        body=draft_body,
        security_events=state.security_events,
        logger=LOGGER,
        attempt_sanitize=True,
    )
    state.output_guard_passed = guard.passed

    print("security demo output")
    print(f"redacted_email_body: {state.redacted_email_body}")
    print(f"detected_language: {detected_language.value}")
    print(f"response_language: {response_language.value}")
    print(f"output_guard_passed: {state.output_guard_passed}")
    print(f"security_events: {state.security_events}")
    print(f"guarded_email_subject: {guard.sanitized_subject}")
    print(f"guarded_email_body: {guard.sanitized_body}")


def run(args: argparse.Namespace) -> int:
    config = AppConfig.from_env(env_file=args.env_file)
    configure_logging(config.log_level)

    LOGGER.info("Bootstrap initialized.")
    if args.demo_security:
        run_security_demo()
        return 0

    if args.scenario is None:
        print("Phase 0 bootstrap ready. Use --scenario once graph execution is implemented.")
    else:
        print(f"Phase 0 bootstrap received scenario={args.scenario}. Execution is not wired yet.")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
