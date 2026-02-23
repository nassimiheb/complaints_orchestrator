"""Web server entrypoint for the complaints orchestrator UI."""

from __future__ import annotations

import argparse
import os

import uvicorn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="complaints-web-ui",
        description="Run the local web UI for the complaints orchestrator.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")
    parser.add_argument("--env-file", default=".env", help="Path to dotenv file.")
    parser.add_argument(
        "--skip-index-check",
        action="store_true",
        help="Skip checking/building the local RAG index on startup.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    os.environ["CCO_ENV_FILE"] = args.env_file
    if args.skip_index_check:
        os.environ["CCO_WEB_SKIP_INDEX_BUILD"] = "1"

    uvicorn.run(
        "complaints_orchestrator.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
