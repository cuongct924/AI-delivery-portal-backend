"""MCP Server for the MLOps Lifecycle Golden Path actions (Train->Track->
Register, Register->Deploy, Serving LLM). Each tool is a thin HTTP client
into orchestration-api's existing `/datasets`, `/trigger-training`,
`/models`, `/deploy-model`, `/llm-deploy` endpoints — no business logic
duplicated here, same convention as llmops-golden-paths-server (that one covers
the LLMOps domain — prompt/RAG lifecycle — this one covers training/
deploy). See README.md for auth requirements.
"""

import os
from typing import Final

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
MCP_SERVER_URL: Final[str] = os.getenv("MCP_SERVER_URL", "http://localhost:9004/mcp")
# Scope required on top of a merely-valid token for the 4 NEEDS_CONFIRMATION
# tools below — provisioning this scope for a given Thunder client is an
# IdP-side step, not something this code can guarantee (see docstrings).
# Distinct from llmops-golden-paths-server's own "golden-paths:mutate" — a caller
# authorized to mutate LLMOps prompt/RAG state is not automatically
# authorized to trigger real training compute or touch production traffic.
MUTATE_SCOPE: Final[str] = "mlops-golden-paths:mutate"

# A valid Thunder Bearer token is required for every tool call once
# token_verifier is set below — closes the same "confused deputy" gap
# llmops-golden-paths-server's server.py closes (see that file's comment for the
# full rationale): a caller hitting this server's streamable-http endpoint
# directly, bypassing orchestration-api/chat.py, cannot execute anything
# with zero verification. `required_scopes` stays empty (any authenticated
# caller may call validate_dataset); the other 4 tools additionally require
# MUTATE_SCOPE, checked in-body since AuthSettings has no per-tool scoping.
mcp = MCPServer(
    "mlops-golden-paths-server",
    auth=AuthSettings(
        issuer_url=AnyHttpUrl(THUNDER_URL), resource_server_url=AnyHttpUrl(MCP_SERVER_URL)
    ),
    token_verifier=ThunderTokenVerifier(),
)

AUTO_EXECUTABLE: Final = ToolAnnotations(read_only_hint=False)
# Mutates live state — chat.py's tool loop must confirm before calling, AND
# (see MUTATE_SCOPE above) the caller's token must carry the scope.
NEEDS_CONFIRMATION: Final = ToolAnnotations(read_only_hint=False, destructive_hint=True)


def _post(path: str, payload: dict) -> dict | list[dict]:
    """Return type is a union because /datasets/validate is the one
    endpoint here returning a JSON array (list[CheckResultResponse]), not
    an object — every other tool's caller gets back a plain dict."""
    response = httpx.post(
        f"{ORCHESTRATION_API_URL}{path}", json=payload, headers=auth_headers(), timeout=30
    )
    response.raise_for_status()
    result: dict | list[dict] = response.json()
    return result


def _require_confirmed_mutation(confirm: bool) -> dict | None:
    """Gate for trigger_training/register_model/prepare_deploy/
    prepare_llm_deploy. Returns an error dict (never raises) when the call
    should be rejected, `None` when it may proceed."""
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
def validate_dataset(
    dataset_uri: str,
    task_type: str,
    target_column: str | None = None,
    time_column: str | None = None,
) -> dict | list[dict]:
    """Run data-quality checks against a dataset before training — read-only,
    doesn't mutate anything. Returns a list of {check_name, severity,
    message, details}."""
    return _post(
        "/datasets/validate",
        {
            "dataset_uri": dataset_uri,
            "task_type": task_type,
            "target_column": target_column,
            "time_column": time_column,
        },
    )


@mcp.tool(annotations=NEEDS_CONFIRMATION)
def trigger_training(
    model_name: str,
    dataset_uri: str,
    task_type: str,
    confirm: bool = False,
    # sklearn by default — "algorithm" only applies to that architecture;
    # mlp/lstm/nlp/cv use the fields below instead.
    architecture: str = "sklearn",
    algorithm: str | None = None,
    target_column: str | None = None,
    id_columns: list[str] | None = None,
    time_column: str | None = None,
    base_model_uri: str | None = None,
    # DL hyperparameters — unused for architecture="sklearn".
    hidden_layers: list[int] | None = None,
    dropout: float | None = None,
    sequence_length: int | None = None,
    num_layers: int | None = None,
    hidden_size: int | None = None,
    learning_rate: float | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    optimizer: str | None = None,
    # BYOC — only used when algorithm="custom".
    code_repo_url: str | None = None,
    entrypoint_path: str | None = None,
    custom_config: str | None = None,
    # HPO — only used when architecture is "mlp"/"lstm" and search_strategy
    # is not "fixed" (the default).
    search_strategy: str | None = None,
    num_trials: int | None = None,
    search_space_json: str | None = None,
    objective_metric: str | None = None,
    objective_direction: str | None = None,
    # NLP — only used when architecture="nlp".
    text_column: str | None = None,
    base_model_name: str | None = None,
) -> dict | list[dict]:
    """Trigger the train-register Argo Workflow — spends real compute.
    Mirrors POST /trigger-training's full parameter set (routers/models.py
    TriggerTrainingRequest) exactly; see that model's field comments for
    which architecture each optional field applies to. Requires `confirm`
    and a token with the mlops-golden-paths:mutate scope.
    """
    error = _require_confirmed_mutation(confirm)
    if error is not None:
        return error
    return _post(
        "/trigger-training",
        {
            "model_name": model_name,
            "dataset_uri": dataset_uri,
            "task_type": task_type,
            "architecture": architecture,
            "algorithm": algorithm,
            "target_column": target_column,
            "id_columns": id_columns,
            "time_column": time_column,
            "base_model_uri": base_model_uri,
            "hidden_layers": hidden_layers,
            "dropout": dropout,
            "sequence_length": sequence_length,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "code_repo_url": code_repo_url,
            "entrypoint_path": entrypoint_path,
            "custom_config": custom_config,
            "search_strategy": search_strategy,
            "num_trials": num_trials,
            "search_space_json": search_space_json,
            "objective_metric": objective_metric,
            "objective_direction": objective_direction,
            "text_column": text_column,
            "base_model_name": base_model_name,
        },
    )


@mcp.tool(annotations=NEEDS_CONFIRMATION)
def register_model(
    name: str,
    artifact_uri: str,
    task_type: str,
    confirm: bool = False,
    dataset_version: str | None = None,
) -> dict | list[dict]:
    """Register a trained model artifact in the Model Registry — mutates
    live state (a new model version becomes queryable). Requires `confirm`
    and a token with the mlops-golden-paths:mutate scope.
    """
    error = _require_confirmed_mutation(confirm)
    if error is not None:
        return error
    return _post(
        "/models/register",
        {
            "name": name,
            "artifact_uri": artifact_uri,
            "task_type": task_type,
            "dataset_version": dataset_version,
        },
    )


@mcp.tool(annotations=NEEDS_CONFIRMATION)
def prepare_deploy(
    model_name: str,
    model_version: str,
    confirm: bool = False,
    # "direct" | "canary" | "ab" | "blue-green".
    traffic_strategy: str = "direct",
    traffic_percent: int | None = None,
    # "pr-gated" | "instant" — "instant" deploys for real immediately.
    release_strategy: str = "pr-gated",
) -> dict | list[dict]:
    """Prepare (and, if release_strategy="instant", immediately apply) a
    model deploy manifest — can change production traffic. Requires
    `confirm` and a token with the mlops-golden-paths:mutate scope.
    """
    error = _require_confirmed_mutation(confirm)
    if error is not None:
        return error
    return _post(
        "/deploy-model/prepare",
        {
            "model_name": model_name,
            "model_version": model_version,
            "traffic_strategy": traffic_strategy,
            "traffic_percent": traffic_percent,
            "release_strategy": release_strategy,
        },
    )


@mcp.tool(annotations=NEEDS_CONFIRMATION)
def prepare_llm_deploy(
    model_name: str,
    huggingface_model_id: str,
    gpu_type: str,
    confirm: bool = False,
    runtime: str = "vllm",
    gpu_count: int = 1,
    quantization: str = "none",
    max_context_length: int = 4096,
    traffic_strategy: str = "direct",
    traffic_percent: int | None = None,
    release_strategy: str = "pr-gated",
) -> dict | list[dict]:
    """Prepare (and, if release_strategy="instant", immediately apply) a
    self-hosted LLM deploy manifest via vLLM/KServe — can change production
    traffic. Requires `confirm` and a token with the
    mlops-golden-paths:mutate scope.
    """
    error = _require_confirmed_mutation(confirm)
    if error is not None:
        return error
    return _post(
        "/llm-deploy/prepare",
        {
            "model_name": model_name,
            "huggingface_model_id": huggingface_model_id,
            "runtime": runtime,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "quantization": quantization,
            "max_context_length": max_context_length,
            "traffic_strategy": traffic_strategy,
            "traffic_percent": traffic_percent,
            "release_strategy": release_strategy,
        },
    )


if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "9004"))
    mcp.run(transport="streamable-http", host=host, port=port)
