"""Prompt Registry API — manages versions of system prompts, kept fully
separate from the Model Registry (unlike traditional MLOps, see docs/architecture.md).
The Portal reads this data through the `plugins/prompt-registry/` plugin
(Backstage), via the `/orchestration-api` proxy declared in app-config.yaml.

Backed by MlflowPromptRegistryAdapter (kind="prompt") — MLflow's native
Prompt Registry (mlflow.genai.*), not the file-backed adapter
routers/rag.py uses for "rag-index" (a prompt is a real MLflow-tracked
artifact; a RAG index pointer isn't). Two personas ("mlops", "k8s") are
seeded at import time so existing behavior survives a first restart
unchanged; new personas register via POST /prompts.
"""

from datetime import datetime

from auth.thunder import get_current_user
from evaluations.gate import evaluate_gate
from evaluations.llm_judge import judge_response
from fastapi import APIRouter, Depends, HTTPException
from observability.dora_metrics import DEPLOYMENT_EVENTS, GATE_EVALUATIONS, INCIDENT_RECOVERY
from pydantic import BaseModel

from adapters.factory import (
    get_eval_result_adapter,
    get_llm_gateway_adapter,
    get_prompt_registry_adapter,
)

router = APIRouter(prefix="/prompts", tags=["prompts"])

llm_gateway_adapter = get_llm_gateway_adapter()
registry_adapter = get_prompt_registry_adapter()
eval_result_adapter = get_eval_result_adapter()


class PromptNamesResponse(BaseModel):
    names: list[str]


class PromptVersionsResponse(BaseModel):
    versions: list[str]


class PromptActiveVersionResponse(BaseModel):
    name: str
    active_version: str | None


class PromptVersion(BaseModel):
    id: str
    name: str
    version: str
    persona: str
    content: str


class DraftPromptRequest(BaseModel):
    name: str
    persona: str
    content: str


class DraftPromptResponse(BaseModel):
    id: str
    name: str
    version: str
    persona: str
    content: str


class PromptEvalCase(BaseModel):
    question: str


class EvaluatePromptRequest(BaseModel):
    version: str
    eval_cases: list[PromptEvalCase]
    # Overridable — same reasoning as routers/rag.py's RagEvaluateRequest.model.
    model: str = "claude-sonnet-5"


class EvaluatePromptResponse(BaseModel):
    passed: bool
    pass_rate: float
    results: list[dict[str, object]]
    # total_cost_usd is None when the model has no cost entry configured.
    total_tokens: int
    total_cost_usd: float | None


class ActivatePromptRequest(BaseModel):
    version: str


class ActivatePromptResponse(BaseModel):
    name: str
    active_version: str


def _seed_default_prompts() -> None:
    defaults = {
        "mlops": {
            "persona": "MLOps Assistant",
            "content": "You are the MLOps assistant for the AI Delivery Portal. You help "
            "ML engineers look up experiments, model registry entries, and deploy "
            "status via MCP tools.",
        },
        "k8s": {
            "persona": "K8s Assistant",
            "content": "You are the Kubernetes operations assistant. You may only read "
            "status (pods, logs, events) via MCP tools — you have no write/delete permission.",
        },
    }
    for name, metadata in defaults.items():
        if registry_adapter.get_active_version("prompt", name) is None:
            version = registry_adapter.register_version("prompt", name, metadata)
            registry_adapter.set_active_version("prompt", name, version)


_seed_default_prompts()


@router.get("", response_model=PromptNamesResponse)
def list_prompt_names(user: dict = Depends(get_current_user)) -> PromptNamesResponse:
    """Every drafted persona key, active or not — backs the Portal's
    promptNamePicker (a user evaluating a draft picks from this list before
    it has an active version)."""
    return PromptNamesResponse(names=registry_adapter.list_names("prompt"))


@router.get("/{name}/versions", response_model=PromptVersionsResponse)
def list_prompt_versions(
    name: str, user: dict = Depends(get_current_user)
) -> PromptVersionsResponse:
    versions = registry_adapter.list_versions("prompt", name)
    return PromptVersionsResponse(versions=sorted(versions, key=int))


@router.get("/{name}/active", response_model=PromptActiveVersionResponse)
def get_prompt_active_version(
    name: str, user: dict = Depends(get_current_user)
) -> PromptActiveVersionResponse:
    # Mirrors routers/rag.py's get_rag_active_version — added for
    # agents/mcp-servers/observability-server's get_active_prompt_version,
    # which used to fetch every persona and filter client-side because no
    # by-name endpoint existed.
    return PromptActiveVersionResponse(
        name=name, active_version=registry_adapter.get_active_version("prompt", name)
    )


@router.get("/{prompt_id}", response_model=PromptVersion)
def get_prompt(prompt_id: str, user: dict = Depends(get_current_user)) -> PromptVersion:
    name, _, version = prompt_id.rpartition("-v")
    if not name:
        raise HTTPException(404, f"Prompt not found: {prompt_id}")
    try:
        metadata = registry_adapter.get_version("prompt", name, version)
    except ValueError as exc:
        raise HTTPException(404, f"Prompt not found: {prompt_id}") from exc
    return PromptVersion(
        id=prompt_id,
        name=name,
        version=version,
        persona=str(metadata["persona"]),
        content=str(metadata["content"]),
    )


@router.post("", response_model=DraftPromptResponse)
def draft_prompt(
    request: DraftPromptRequest, user: dict = Depends(get_current_user)
) -> DraftPromptResponse:
    version = registry_adapter.register_version(
        "prompt", request.name, {"persona": request.persona, "content": request.content}
    )
    return DraftPromptResponse(
        id=f"{request.name}-v{version}",
        name=request.name,
        version=version,
        persona=request.persona,
        content=request.content,
    )


@router.post("/{name}/evaluate", response_model=EvaluatePromptResponse)
def evaluate_prompt(
    name: str, request: EvaluatePromptRequest, user: dict = Depends(get_current_user)
) -> EvaluatePromptResponse:
    metadata = registry_adapter.get_version("prompt", name, request.version)
    system_prompt = metadata["content"]

    results: list[dict[str, object]] = []
    total_tokens = 0
    total_cost_usd = 0.0
    cost_known = True
    passed_count = 0
    for eval_case in request.eval_cases:
        response = llm_gateway_adapter.chat_completion(
            model=request.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": eval_case.question},
            ],
        )
        answer = response["choices"][0]["message"]["content"]
        total_tokens += (response.get("usage") or {}).get("total_tokens", 0)
        response_cost = response.get("response_cost_usd")
        if response_cost is None:
            cost_known = False
        else:
            total_cost_usd += response_cost
        judge_result = judge_response(eval_case.question, answer)
        gate_result = evaluate_gate(judge_result)
        results.append(
            {"question": eval_case.question, "answer": answer, "passed": gate_result["passed"]}
        )
        if gate_result["passed"]:
            passed_count += 1

        # Persist judge result for MTTR calculation
        eval_result_adapter.log_judge_result(
            kind="prompt",
            name=name,
            version=request.version,
            judge_result=judge_result,
            passed=gate_result["passed"],
        )

    pass_rate = passed_count / len(results) if results else 0.0
    overall_passed = pass_rate >= 0.8

    # Emit DORA gate evaluation metric (LLMOps track, prompt)
    GATE_EVALUATIONS.labels(
        track="llmops",
        subject_type="prompt",
        subject_id=f"{name}:{request.version}",
        passed=str(overall_passed).lower(),
    ).inc()

    return EvaluatePromptResponse(
        passed=overall_passed,
        pass_rate=pass_rate,
        results=results,
        total_tokens=total_tokens,
        total_cost_usd=total_cost_usd if cost_known else None,
    )


@router.post("/{name}/activate", response_model=ActivatePromptResponse)
def activate_prompt(
    name: str, request: ActivatePromptRequest, user: dict = Depends(get_current_user)
) -> ActivatePromptResponse:
    registry_adapter.set_active_version("prompt", name, request.version)

    # Emit DORA deployment event (LLMOps track)
    DEPLOYMENT_EVENTS.labels(
        track="llmops",
        subject_type="prompt",
        subject_id=name,
        event_type="deploy",
    ).inc()

    # MTTR: find last judge failure for this prompt and calculate recovery time
    last_failure = eval_result_adapter.get_last_failure_at("prompt", name)
    if last_failure is not None:
        recovery_seconds = (datetime.now() - last_failure).total_seconds()
        INCIDENT_RECOVERY.labels(
            track="llmops",
            subject_type="prompt",
            subject_id=name,
        ).observe(recovery_seconds)

    return ActivatePromptResponse(name=name, active_version=request.version)
