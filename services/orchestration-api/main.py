"""FastAPI BFF — acts as an MCP Client, routing requests from the Portal UI
to the AI LLM (Claude, or any model registered in
infra/llm-gateways/litellm-config.yaml)."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from mcp_client import McpToolRegistry
from prometheus_fastapi_instrumentator import Instrumentator
from routers import (
    chat,
    llm_serving,
    models,
    monitoring,
    portal_assistant,
    prompts,
    rag,
    recommendations,
)

# Without this, app-level logger.info() calls are silently dropped —
# uvicorn only configures its own loggers.
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    registry = McpToolRegistry()
    await registry.connect_all()  # never raises; degrades gracefully
    app.state.mcp_registry = registry
    yield
    await registry.aclose()


app = FastAPI(title="AI Delivery Portal — Orchestration API", lifespan=lifespan)
app.include_router(chat.router)
app.include_router(prompts.router)
app.include_router(models.router)
app.include_router(recommendations.router)
app.include_router(monitoring.router)
app.include_router(llm_serving.router)
app.include_router(rag.router)
app.include_router(portal_assistant.router)

# Expose /metrics — scraped by Prometheus (infra/monitoring/prometheus.yml)
Instrumentator().instrument(app).expose(app)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}
