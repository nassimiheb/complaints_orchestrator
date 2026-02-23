"""FastAPI app exposing a simple web interface for the orchestrator."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from complaints_orchestrator.web.schemas import RunCaseRequest, RunCaseResponse, ScenarioPreview
from complaints_orchestrator.web.service import (
    WebRuntime,
    initialize_runtime,
    load_scenario_previews,
    run_case,
)

LOGGER = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


def _resolve_env_file(env_file: str | None) -> str:
    if env_file is not None:
        return env_file
    return os.getenv("CCO_ENV_FILE", ".env")


def _resolve_index_build_behavior(ensure_index_if_missing: bool | None) -> bool:
    if ensure_index_if_missing is not None:
        return ensure_index_if_missing
    skip_flag = os.getenv("CCO_WEB_SKIP_INDEX_BUILD", "").strip().lower()
    return skip_flag not in {"1", "true", "yes"}


def _get_runtime(app: FastAPI) -> WebRuntime:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("Web runtime is not initialized.")
    return runtime


def create_app(
    *,
    env_file: str | None = None,
    scenarios_file: Path | None = None,
    runtime: WebRuntime | None = None,
    ensure_index_if_missing: bool | None = None,
) -> FastAPI:
    """Create a configured FastAPI app instance."""

    selected_env_file = _resolve_env_file(env_file)
    should_ensure_index = _resolve_index_build_behavior(ensure_index_if_missing)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if runtime is not None:
            app.state.runtime = runtime
            yield
            return

        app.state.runtime = initialize_runtime(
            env_file=selected_env_file,
            ensure_index_if_missing=should_ensure_index,
        )
        yield

    app = FastAPI(
        title="Complaints Orchestrator Web UI",
        description="Web UI and API wrapper around the LangGraph complaints workflow.",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "app_title": "Customer Complaints Orchestrator",
            },
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/scenarios", response_model=list[ScenarioPreview])
    def list_scenarios() -> list[ScenarioPreview]:
        try:
            return load_scenario_previews(scenarios_file=scenarios_file)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Failed to load scenarios for UI.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/cases/run", response_model=RunCaseResponse)
    def run_case_endpoint(payload: RunCaseRequest, request: Request) -> RunCaseResponse:
        try:
            runtime_ctx = _get_runtime(request.app)
            return run_case(request_payload=payload, runtime=runtime_ctx)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Case execution failed.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
