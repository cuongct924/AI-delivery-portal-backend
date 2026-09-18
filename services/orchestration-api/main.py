"""FastAPI BFF — acts as an MCP Client, routing requests from the Portal UI
to the AI LLM (Claude, or any model registered in
infra/llm-gateways/litellm-config.yaml)."""

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp_client import McpToolRegistry
from observability.dora_metrics import LLM_SPEND_USD
from prometheus_fastapi_instrumentator import Instrumentator
from routers import (
    chat,
    golden_paths,
    llm_models,
    llm_serving,
    mock_observer,
    models,
    monitoring,
    portal_assistant,
    prompts,
    rag,
)

from adapters.factory import get_llm_gateway_adapter

# Without this, app-level logger.info() calls are silently dropped —
# uvicorn only configures its own loggers.
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


async def _refresh_llm_spend() -> None:
    """Background task to refresh LLM spend metrics every 5 minutes."""
    adapter = get_llm_gateway_adapter()
    while True:
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            spend_report = adapter.get_spend_report(start_date, end_date, group_by="model")
            for entry in spend_report:
                model = entry.get("model")
                cost = entry.get("cost", 0.0)
                if model and cost is not None:
                    LLM_SPEND_USD.labels(model=model).set(float(cost))  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 — background task must not crash the app
            logger.warning("Failed to refresh LLM spend metrics: %s", exc)
        await asyncio.sleep(300)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    registry = McpToolRegistry()
    await registry.connect_all()  # never raises; degrades gracefully
    app.state.mcp_registry = registry

    # Start background task for LLM spend metrics
    spend_task = asyncio.create_task(_refresh_llm_spend())

    yield

    spend_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await spend_task

    await registry.aclose()


app = FastAPI(title="AI Delivery Portal — Orchestration API", lifespan=lifespan)

# The Portal frontend calls the mock observer (routers/mock_observer.py)
# directly from the browser, so it needs CORS. Comma-separated override via
# CORS_ALLOWED_ORIGINS; defaults to the local Backstage dev server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:7007"
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(prompts.router)
app.include_router(models.router)
app.include_router(monitoring.router)
app.include_router(llm_serving.router)
app.include_router(rag.router)
app.include_router(portal_assistant.router)
app.include_router(golden_paths.router)
app.include_router(llm_models.router)
app.include_router(mock_observer.router)

# feast (via adapters.factory) sets PROMETHEUS_MULTIPROC_DIR at import time,
# which makes Instrumentator's /metrics read empty multiprocess .db files
# this app never writes — pop it so /metrics stays in-process.
os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)

# Expose /metrics — scraped by Prometheus (infra/monitoring/prometheus.yml)
Instrumentator().instrument(app).expose(app)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}
