"""MCP Server for the "how do I run Golden Path X" domain — read-only,
thin HTTP client into orchestration-api's /golden-paths endpoints (no
business logic duplicated here, same convention as golden-paths-server).

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


if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "9003"))
    mcp.run(transport="streamable-http", host=host, port=port)
