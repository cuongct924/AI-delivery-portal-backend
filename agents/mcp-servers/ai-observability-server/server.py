"""MCP Server for the AI/ML/LLM health domain — read-only.

Infra-level tools (k8s pod/logs, raw PromQL, OpenChoreo promotion state) are
left to OpenChoreo's own Control Plane / Observability Plane MCP servers; this
server only covers what OpenChoreo has no concept of: MLflow registry/metrics,
model latency, LiteLLM spend, and orchestration-api's prompt/RAG active
versions. See agents/mcp-servers/llmops-golden-paths-server/ and
agents/mcp-servers/mlops-golden-paths-server/ for the write-side domains.
"""

import os
from typing import Final, TypedDict

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from adapters.ai_platform.interfaces import ModelSummary
from adapters.ai_platform.llm_gateway_adapter import LiteLLMGatewayAdapter
from adapters.ai_platform.mlflow_adapter import MlflowAdapter

mcp = MCPServer("ai-observability-server")
adapter = MlflowAdapter()
llm_gateway_adapter = LiteLLMGatewayAdapter()
PROMETHEUS_URL: Final[str] = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
# orchestration-api owns prompt/RAG active-version state
# (services/orchestration-api/.state/llmops-registry.json, via
# JsonFileVersionRegistryAdapter) — this server calls it over HTTP rather
# than reading that file directly. It runs in its own container with no
# shared filesystem (see docker-compose.yml), and every other real
# integration here (MLflow, Prometheus, LiteLLM) is already an HTTP call
# to another service, not a shared file — same pattern, not a new one.
ORCHESTRATION_API_URL: Final[str] = os.getenv("ORCHESTRATION_API_URL", "http://localhost:8000")

READ_ONLY: Final = ToolAnnotations(read_only_hint=True)


class LatencyCheck(TypedDict):
    model: str
    namespace: str
    p95_latency_ms: float | None
    threshold_ms: float
    breached: bool


class ActiveVersion(TypedDict):
    name: str
    active_version: str | None


class EvalScoreTrend(TypedDict):
    name: str
    kind: str
    scores: list[dict[str, object]]
    note: str


@mcp.tool(annotations=READ_ONLY)
def list_experiments() -> list[ModelSummary]:
    """List models registered in the MLflow Registry."""
    return adapter.list_models()


@mcp.tool(annotations=READ_ONLY)
def get_model_metrics(name: str, version: str) -> dict[str, float]:
    """Get metrics (accuracy, f1, ...) for a specific model version."""
    return adapter.get_model_metrics(name, version)


@mcp.tool(annotations=READ_ONLY)
def check_model_latency(model_name: str, namespace: str, threshold_ms: float = 500) -> LatencyCheck:
    """Check whether a model's p95 latency exceeds a threshold (assumes the
    `model_inference_duration_ms` metric is exposed by KServe/BentoML, labeled by model
    and namespace). `namespace` is required, not optional — with
    infra/environments/{dev,staging,prod}/inference-services/{mlops-team,llmops-team}/,
    the same model_name can be deployed in multiple namespaces
    (ai-delivery-portal-<env>-<tenant>) at once, so a query with no namespace
    filter would silently aggregate across all of them."""
    promql = (
        "histogram_quantile(0.95, sum(rate("
        f'model_inference_duration_ms_bucket{{model="{model_name}", namespace="{namespace}"}}'
        "[5m])) by (le))"
    )
    response = httpx.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": promql}, timeout=10)
    response.raise_for_status()
    result = response.json()
    data = result.get("data")
    values = data.get("result", []) if isinstance(data, dict) else []
    p95 = float(values[0]["value"][1]) if values else None
    return {
        "model": model_name,
        "namespace": namespace,
        "p95_latency_ms": p95,
        "threshold_ms": threshold_ms,
        "breached": p95 is not None and p95 > threshold_ms,
    }


@mcp.tool(annotations=READ_ONLY)
def get_llm_spend(
    start_date: str, end_date: str, group_by: str | None = None
) -> list[dict[str, object]]:
    """LLM spend/cost over [start_date, end_date] (YYYY-MM-DD), from
    LiteLLM's real spend ledger (GET /global/spend/report — not a mock).
    group_by: "team", "customer", or omit for per-api-key totals."""
    return llm_gateway_adapter.get_spend_report(start_date, end_date, group_by)


@mcp.tool(annotations=READ_ONLY)
def get_active_prompt_version(name: str) -> ActiveVersion:
    """Which version of a system prompt is currently active. LLMOps
    releases are Instant-only, not PR-gated (docs/llmops-lifecycle-plan.md
    mục Q4) — there is no Git/ArgoCD trail to read this from, unlike a
    model deploy, so this calls orchestration-api's own registry directly.
    Mirrors get_active_rag_version below, which calls the "rag-index"
    equivalent (GET /rag/{collection})."""
    response = httpx.get(f"{ORCHESTRATION_API_URL}/prompts/{name}/active", timeout=10)
    response.raise_for_status()
    data = response.json()
    return {"name": name, "active_version": data["active_version"]}


@mcp.tool(annotations=READ_ONLY)
def get_active_rag_version(collection: str) -> ActiveVersion:
    """Which RAG index version is currently active for a collection. Same
    Instant-only reasoning as get_active_prompt_version — calls
    orchestration-api's GET /rag/{collection} (added alongside this tool;
    no read endpoint existed for RAG's active version before)."""
    response = httpx.get(f"{ORCHESTRATION_API_URL}/rag/{collection}", timeout=10)
    response.raise_for_status()
    data = response.json()
    return {"name": collection, "active_version": data["active_version"]}


@mcp.tool(annotations=READ_ONLY)
def get_eval_score_trend(name: str, kind: str = "prompt") -> EvalScoreTrend:
    """Score history (safety/correctness/relevance) for a prompt or
    RAG index over time. (mock — evaluations/llm_judge.py's judge_response()
    computes a score per call but nothing persists it anywhere today; a
    real version would need evaluate_prompt/rag_evaluate in
    routers/prompts.py|rag.py to write each result into the registry
    first, which they currently don't)."""
    return {
        "name": name,
        "kind": kind,
        "scores": [],
        "note": "mock data — judge scores aren't persisted anywhere yet, see this tool's docstring",
    }


if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "9001"))
    mcp.run(transport="streamable-http", host=host, port=port)
