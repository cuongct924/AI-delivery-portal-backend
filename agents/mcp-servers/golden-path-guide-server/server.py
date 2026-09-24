"""MCP Server for the "how do I run Golden Path X" domain — read-only,
thin HTTP client into orchestration-api's /golden-paths endpoints (no
business logic duplicated here, same convention as llmops-golden-paths-server).

Deliberately not RAG: the mlops/llmops Golden Path templates are a small,
fixed set (~8), each with a short structured spec (parameters + steps) —
reading the live Backstage Catalog entity on every call is simpler and
never goes stale, unlike a RAG index that needs re-ingesting after every
template.yaml edit.
"""

import os
from typing import Final, TypedDict

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

mcp = MCPServer("golden-path-guide-server")
ORCHESTRATION_API_URL: Final[str] = os.getenv("ORCHESTRATION_API_URL", "http://localhost:8000")

READ_ONLY: Final = ToolAnnotations(read_only_hint=True)


class GoldenPathSummary(TypedDict):
    name: str
    title: str
    description: str
    tags: list[str]


class GoldenPathParameter(TypedDict):
    name: str
    title: str
    description: str
    required: bool


class GoldenPathStep(TypedDict):
    name: str
    action: str


class GoldenPathGuide(TypedDict):
    name: str
    title: str
    description: str
    tags: list[str]
    parameters: list[GoldenPathParameter]
    steps: list[GoldenPathStep]


class CostEstimate(TypedDict):
    golden_path: str
    stage: str
    estimated_cost: float
    currency: str
    breakdown: dict[str, float]


@mcp.tool(annotations=READ_ONLY)
def list_golden_paths() -> list[GoldenPathSummary]:
    """List every mlops/llmops Golden Path Scaffolder template — name,
    title, description, tags. Use this first to find the right template's
    `name` before calling get_golden_path_guide."""
    response = httpx.get(f"{ORCHESTRATION_API_URL}/golden-paths", timeout=10)
    response.raise_for_status()
    return response.json()


@mcp.tool(annotations=READ_ONLY)
def get_golden_path_guide(name: str) -> GoldenPathGuide:
    """Full "how to run" guide for one Golden Path template: every input
    parameter (with description and whether it's required) and the
    ordered list of steps it executes. `name` is the value returned by
    list_golden_paths, e.g. "train-track-register"."""
    response = httpx.get(f"{ORCHESTRATION_API_URL}/golden-paths/{name}", timeout=10)
    response.raise_for_status()
    return response.json()


@mcp.tool(annotations=READ_ONLY)
def get_golden_path_schema(name: str) -> list[dict[str, object]]:
    """The raw JSON schema of one Golden Path template's input parameters —
    types, enums, defaults and allOf/if branching. Use this (not
    get_golden_path_guide, which flattens the fields) when you need to fill
    the form: it tells you exactly which fields exist per branch. `name` is
    a value from list_golden_paths, e.g. "train-track-register"."""
    response = httpx.get(f"{ORCHESTRATION_API_URL}/golden-paths/{name}/schema", timeout=10)
    response.raise_for_status()
    return response.json()


@mcp.tool(annotations=READ_ONLY)
def propose_golden_path_draft(name: str, values: dict[str, object]) -> dict[str, object]:
    """Validate a set of form values you filled for a Golden Path template
    against its live schema. Pure — nothing is submitted. Returns
    {ok, form_data, missing, errors}: `missing` lists required fields you
    left out (fill them and call again), `errors` lists schema violations.
    `name` is a value from list_golden_paths; `values` is a flat object of
    field name -> value."""
    response = httpx.post(
        f"{ORCHESTRATION_API_URL}/golden-paths/{name}/draft",
        json={"values": values},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


@mcp.tool(annotations=READ_ONLY)
def estimate_golden_path_cost(
    golden_path: str,
    stage: str,
    params: dict[str, object] | None = None,
) -> CostEstimate:
    """Pre-flight cost estimate (USD) for running a Golden Path, so an agent can
    answer "how much will this cost". `golden_path` is a name from
    list_golden_paths; `stage` is build | gate | run; `params` are the path's
    own cost drivers, e.g. {"gpuType": "H100", "gpuCount": 2} for
    llm-serve-deploy or {"epochs": 10} for train-track-register."""
    response = httpx.post(
        f"{ORCHESTRATION_API_URL}/costs/estimate",
        json={"golden_path": golden_path, "stage": stage, "params": params or {}},
        timeout=10,
    )
    response.raise_for_status()
    body = response.json()
    return {
        "golden_path": golden_path,
        "stage": stage,
        "estimated_cost": body["estimated_cost"],
        "currency": body.get("currency", "USD"),
        "breakdown": body.get("breakdown", {}),
    }


if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "9003"))
    mcp.run(transport="streamable-http", host=host, port=port)
