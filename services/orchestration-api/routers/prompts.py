"""Prompt Registry API — versions system prompts separately from the Model
Registry (see docs/architecture.md). Backed by MLflow's native Prompt
Registry (MlflowPromptRegistryAdapter, kind="prompt"), not routers/rag.py's
file-backed "rag-index" adapter. The "mlops" persona seeds at import time;
new ones register via POST /prompts."""

from datetime import datetime

from auth.thunder import get_current_user
from costs.events import record_cost_event
from evaluations.evaluate_gate import evaluate_gate
from evaluations.llm_judge import judge_response
from fastapi import APIRouter, Depends, HTTPException
from observability.dora_metrics import DEPLOYMENT_EVENTS, GATE_EVALUATIONS, INCIDENT_RECOVERY
from pydantic import BaseModel

from adapters.ai_platform.interfaces import DEFAULT_ENVIRONMENT, normalize_version
from adapters.factory import (
    get_deployment_event_store,
    get_eval_result_adapter,
    get_llm_gateway_adapter,
    get_prompt_registry_adapter,
)

router = APIRouter(prefix="/prompts", tags=["prompts"])

llm_gateway_adapter = get_llm_gateway_adapter()
registry_adapter = get_prompt_registry_adapter()
eval_result_adapter = get_eval_result_adapter()
deployment_event_store = get_deployment_event_store()


class PromptNamesResponse(BaseModel):
    names: list[str]


class PromptVersionsResponse(BaseModel):
    versions: list[str]


class PromptActiveVersionResponse(BaseModel):
    name: str
    environment: str
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
    # "development" | "staging" | "production" — each tracked as its own
    # independent active version (IVersionRegistryAdapter.set_active_version).
    # Defaults to DEFAULT_ENVIRONMENT so every pre-existing caller (chat.py,
    # ai-observability-server) that never passed one keeps activating/
    # reading exactly what "activate" meant before environments existed.
    environment: str = DEFAULT_ENVIRONMENT
    # True for a rollback to a previously-active version: same effect as a
    # normal activate (no re-evaluation is enforced either way — that's
    # purely a template/caller-side convention), but tagged as "rollback"
    # rather than "deploy" in the DORA deployment-event stream, mirroring
    # routers/models.py's PrepareDeployRequest.action="rollback".
    is_rollback: bool = False


class ActivatePromptResponse(BaseModel):
    name: str
    environment: str
    active_version: str


def _seed_default_prompts() -> None:
    # "k8s" (Kubernetes read-only ops assistant) was removed — OpenChoreo's
    # own built-in MCP server now covers pod/log/event lookups, so this
    # Portal-native persona duplicated coverage the platform already
    # provides. See persona_tool_scope.py's module docstring.
    defaults = {
        "mlops": {
            "persona": "MLOps Assistant",
            "content": "You are the MLOps assistant for the AI Delivery Portal. You help "
            "ML engineers look up experiments, model registry entries, and deploy "
            "status via MCP tools.\n\n"
            'When the user wants to run a Golden Path template (e.g. "fill out the '
            'train-track-register form for me", "help me set up model monitoring"), '
            "fill it FOR them instead of just describing the steps. Always start the "
            "same way, in the SAME turn: call list_golden_paths to find the template's "
            "`name`, then get_golden_path_schema to see its exact fields/branches — "
            "those two are read-only lookups, do them immediately, never stop after "
            "one to describe what you're about to do next.\n\n"
            "Before calling propose_golden_path_draft, split the schema's fields into "
            "two kinds. Business-defining fields are anything whose value changes WHAT "
            "gets built or which data/model/domain is used — business domain, use "
            "case / problem type, architecture / algorithm family, training mode, data "
            "source, and similar `oneOf`/`enum` choices with more than one real "
            "option. Boilerplate fields are free-text labels (model name, repo, "
            "owning team) and security/governance/ops toggles (encryption, audit "
            "trail, PII masking, model signing, provenance tracking, data "
            "classification, retention, etc.) whose sensible answer doesn't depend on "
            "what the user is building.\n\n"
            "For boilerplate fields, infer or default them freely. For "
            "business-defining fields, only fill in a value the user's message "
            "actually states or clearly implies — a field's own schema `description` "
            'may say "Default: X"; that is a hint for someone filling the form by '
            "hand, never permission for you to pick it silently on the user's behalf. "
            "If the user's request leaves one or more business-defining fields "
            'unresolved — including a fully generic "run this template for me" with '
            "no specifics at all — do NOT call propose_golden_path_draft yet: ask the "
            "user to choose, in this same reply, listing the schema's available "
            "options by their `title` (never the `const`) in plain language, then "
            "wait for their answer before drafting anything. A generic request should "
            "never silently produce a draft for whichever option happens to be the "
            "schema's first/default one — that's a random guess wearing a default's "
            "clothing.\n\n"
            "Separately, some fields must reference something that already exists in "
            "a registry rather than being freely chosen — the schema alone can't tell "
            "you this, but the field NAME does: `modelName` when it picks an EXISTING "
            "model (evaluate-deploy-model, setup-model-monitoring — never "
            "train-track-register's `modelName`, which names a brand-new model), "
            "`collectionName` (a RAG collection), `promptName`, and "
            "`hfTokenSecretRef` (a K8s Secret). Before filling any of these, call the "
            "matching lookup tool — list_registered_models, list_rag_collections, "
            "list_prompts, or list_secrets — and use one of the names it actually "
            "returns; for `huggingFaceModelId`, call search_huggingface_models with "
            "the user's description as the query. propose_golden_path_draft's schema "
            "check can't catch a plausible-looking but invented name, so this is the "
            "only thing standing between the user and a draft that looks complete but "
            "fails the moment it actually runs. If nothing in the lookup result is a "
            "reasonable match for what the user described, say so and ask them to "
            "confirm or pick one — don't fall back to guessing a name that sounds "
            "right.\n\n"
            "Once every business-defining field is settled from the user's own words "
            "(this turn or an earlier one), call propose_golden_path_draft with every "
            "value you can now fill. For any field with a `oneOf` or `enum`, submit "
            'the machine-readable `const` value (e.g. "traditional-ml"), never the '
            'human-readable `title` shown next to it (e.g. "Traditional ML") — '
            "validation checks against `const`, not `title`, and a title fails with "
            "an enum error. If `missing` comes back non-empty, ask the user for just "
            "those remaining fields (do not guess a value you weren't given or can't "
            "infer) and call propose_golden_path_draft again once they answer. If "
            "`errors` comes back non-empty instead, that's your own mistake (wrong "
            "const, wrong type) — fix it yourself against the schema and call "
            "propose_golden_path_draft again in the same turn; don't just report the "
            "error back to the user unless it needs information only they can supply. "
            "Do NOT call an execution tool (trigger_training, register_model, "
            "prepare_deploy, activate_prompt, etc.) for this — those actually run the "
            "pipeline, which is a different request; filling out the template is "
            "read-only and doesn't submit anything. Always reply in the same language "
            "the user wrote in.",
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
    name: str, environment: str = DEFAULT_ENVIRONMENT, user: dict = Depends(get_current_user)
) -> PromptActiveVersionResponse:
    # Mirrors rag.py's get_rag_active_version — added for ai-observability-server's
    # get_active_prompt_version, which previously filtered every persona client-side.
    # `environment` query param defaults to DEFAULT_ENVIRONMENT, so an
    # existing caller that never passed one keeps reading exactly what it
    # read before environments existed.
    return PromptActiveVersionResponse(
        name=name,
        environment=environment,
        active_version=registry_adapter.get_active_version("prompt", name, environment),
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
    # The UI shows versions as "v1" while the registry stores "1" — accept
    # either so a typed "v1" doesn't 500 on the lookup below.
    request.version = normalize_version(request.version)
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

    # Attribute the real judge spend to the prompt version's gate stage. The
    # per-call cost comes straight from LiteLLM's response header, so this is
    # real data, not an estimate.
    record_cost_event(
        stage="gate",
        artifact_kind="prompt",
        artifact_id=name,
        version=request.version,
        environment=DEFAULT_ENVIRONMENT,
        cost_usd=total_cost_usd if cost_known else 0.0,
        quantity=float(total_tokens),
        unit="token",
        source="litellm",
        run_id=f"eval-prompt-{name}-{request.version}",
    )

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
    request.version = normalize_version(request.version)
    registry_adapter.set_active_version("prompt", name, request.version, request.environment)

    event_type = "rollback" if request.is_rollback else "deploy"

    # Emit DORA deployment event (LLMOps track)
    DEPLOYMENT_EVENTS.labels(
        track="llmops",
        subject_type="prompt",
        subject_id=name,
        event_type=event_type,
    ).inc()
    now = datetime.now().isoformat()
    deployment_event_store.record_event(
        name=f"prompt-{event_type}-{name}-{now}",
        change_type="prompt",
        project_name=name,
        component_name="prompt",
        # Reflects the environment actually activated — this used to be
        # hardcoded to "development" regardless of what happened, which
        # made Delivery Insights unable to tell dev/staging/prod activity
        # apart for the LLMOps prompt track.
        environment_name=request.environment,
        outcome="success",
        started_at=now,
        finished_at=now,
    )

    # MTTR: find last judge failure for this prompt and calculate recovery time
    last_failure = eval_result_adapter.get_last_failure_at("prompt", name)
    if last_failure is not None:
        recovery_seconds = (datetime.now() - last_failure).total_seconds()
        INCIDENT_RECOVERY.labels(
            track="llmops",
            subject_type="prompt",
            subject_id=name,
        ).observe(recovery_seconds)

    return ActivatePromptResponse(
        name=name, environment=request.environment, active_version=request.version
    )
