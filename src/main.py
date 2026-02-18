"""CLI entrypoint for the complaints orchestrator."""

from __future__ import annotations

import argparse
import logging

from complaints_orchestrator.config import AppConfig
from complaints_orchestrator.logging_config import configure_logging

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
    return parser


def run(args: argparse.Namespace) -> int:
    config = AppConfig.from_env(env_file=args.env_file)
    configure_logging(config.log_level)

    LOGGER.info("Bootstrap initialized.")
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
