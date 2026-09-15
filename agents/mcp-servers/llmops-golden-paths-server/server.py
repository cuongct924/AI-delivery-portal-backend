"""MCP Server for the LLMOps Lifecycle Golden Path actions. Each tool is a
thin HTTP client into orchestration-api's existing endpoints — no business
logic duplicated here. See README.md for auth requirements.
"""

import os
from typing import Final, TypedDict

import httpx
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import AnyHttpUrl
from thunder_client import THUNDER_URL, auth_headers
from token_verifier import ThunderTokenVerifier

ORCHESTRATION_API_URL: Final[str] = os.getenv("ORCHESTRATION_API_URL", "http://localhost:8000")
# Real Catalog-registered address (docker-compose service name) — required by
# AuthSettings as this server's own resource identifier, distinct from the
# host/port it binds to.
MCP_SERVER_URL: Final[str] = os.getenv("MCP_SERVER_URL", "http://localhost:9002/mcp")
# Scope required on top of a merely-valid token for activate_prompt/rag_activate
# — provisioning this scope for a given Thunder client is an IdP-side step, not
# something this code can guarantee (see docstring on those two tools).
MUTATE_SCOPE: Final[str] = "llmops-golden-paths:mutate"

# A valid Thunder Bearer token is required for every tool call once
# token_verifier is set below — this closes the "confused deputy" gap where a
# caller hitting this server's streamable-http endpoint directly (bypassing
# orchestration-api/chat.py) could execute tools with zero verification.
# `required_scopes` stays empty here (any authenticated caller may call the
# AUTO_EXECUTABLE tools); activate_prompt/rag_activate additionally require
# MUTATE_SCOPE, checked in-body since AuthSettings has no per-tool scoping.
mcp = MCPServer(
    "llmops-golden-paths-server",
    auth=AuthSettings(
        issuer_url=AnyHttpUrl(THUNDER_URL), resource_server_url=AnyHttpUrl(MCP_SERVER_URL)
    ),
    token_verifier=ThunderTokenVerifier(),
)

AUTO_EXECUTABLE: Final = ToolAnnotations(read_only_hint=False)
# Mutates live state — chat.py's tool loop must confirm before calling, AND
# (see MUTATE_SCOPE above) the caller's token must carry the scope.
NEEDS_CONFIRMATION: Final = ToolAnnotations(read_only_hint=False, destructive_hint=True)


class EvalCase(TypedDict):
    question: str


def _post(path: str, payload: dict) -> dict:
    response = httpx.post(
        f"{ORCHESTRATION_API_URL}{path}", json=payload, headers=auth_headers(), timeout=30
    )
    response.raise_for_status()
    return response.json()


def _require_confirmed_mutation(confirm: bool) -> dict | None:
    """Gate for activate_prompt/rag_activate. Returns an error dict (never
    raises) when the call should be rejected, `None` when it may proceed."""
    if not confirm:
        return {
            "error": "confirmation required",
            "hint": "resubmit with confirm=True after explicit user approval",
        }
    access_token = get_access_token()
    if access_token is None or MUTATE_SCOPE not in access_token.scopes:
        return {"error": f"token missing required scope {MUTATE_SCOPE!r}"}
    return None


@mcp.tool(annotations=AUTO_EXECUTABLE)
def draft_prompt(name: str, persona: str, content: str) -> dict:
    """Register a new (inactive) prompt version."""
    return _post("/prompts", {"name": name, "persona": persona, "content": content})


@mcp.tool(annotations=AUTO_EXECUTABLE)
def evaluate_prompt(
    name: str, version: str, eval_cases: list[EvalCase], model: str = "claude-sonnet-5"
) -> dict:
    """Run the LLM-as-judge Evaluate Gate against a prompt version and report the pass rate."""
    return _post(
        f"/prompts/{name}/evaluate",
        {"version": version, "eval_cases": eval_cases, "model": model},
    )


@mcp.tool(annotations=NEEDS_CONFIRMATION)
def activate_prompt(name: str, version: str, confirm: bool = False) -> dict:
    """Activate a prompt version for the chat endpoint — mutates live state.

    `confirm` must be explicitly True (chat.py only ever sets it after the
    human approves the pending-confirmation prompt — never from raw model
    output) and the caller's token must carry MUTATE_SCOPE.
    """
    error = _require_confirmed_mutation(confirm)
    if error is not None:
        return error
    return _post(f"/prompts/{name}/activate", {"version": version})


@mcp.tool(annotations=AUTO_EXECUTABLE)
def rag_ingest(
    collection: str, source_paths: list[str], chunk_size: int = 800, chunk_overlap: int = 100
) -> dict:
    """Chunk and embed documents into a Qdrant collection.

    Registers a new (inactive) RAG index version.
    """
    return _post(
        "/rag/ingest",
        {
            "collection": collection,
            "source_paths": source_paths,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    )


@mcp.tool(annotations=AUTO_EXECUTABLE)
def rag_evaluate(
    collection: str,
    index_version: str,
    eval_cases: list[EvalCase],
    model: str = "claude-sonnet-5",
) -> dict:
    """Run the LLM-as-judge Evaluate Gate against a RAG index version and report the pass rate."""
    return _post(
        "/rag/evaluate",
        {
            "collection": collection,
            "index_version": index_version,
            "eval_cases": eval_cases,
            "model": model,
        },
    )


@mcp.tool(annotations=NEEDS_CONFIRMATION)
def rag_activate(collection: str, index_version: str, confirm: bool = False) -> dict:
    """Activate a RAG index version for the chat endpoint — mutates live state.

    See `activate_prompt`'s docstring — same `confirm` + MUTATE_SCOPE gate.
    """
    error = _require_confirmed_mutation(confirm)
    if error is not None:
        return error
    return _post("/rag/activate", {"collection": collection, "index_version": index_version})


if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "9002"))
    mcp.run(transport="streamable-http", host=host, port=port)
