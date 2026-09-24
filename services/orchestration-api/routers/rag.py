"""RAG (Retrieval-Augmented Generation) API — ingest → evaluate → activate,
the RAG-index half of the LLMOps lifecycle (docs/llmops-lifecycle-plan.md).
The prompt-versioning half lives in routers/prompts.py; both use the same
IVersionRegistryAdapter interface, but a different backend: prompts use
MLflow's Prompt Registry, a RAG index uses Qdrant (QdrantVersionRegistryAdapter)
so its version lives next to its embeddings.

`Depends(get_current_user)` on all 3 routes — called from Backstage Custom
Scaffolder Actions, same as every route in models.py except
`/models/register`.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Final

from auth.thunder import get_current_user
from core.config import settings
from costs.events import record_cost_event
from costs.pricing import EMBED_USD_PER_1K_CHUNK
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
    get_registry_adapter,
    get_vector_store_adapter,
)

router = APIRouter(prefix="/rag", tags=["rag"])

llm_gateway_adapter = get_llm_gateway_adapter()
vector_store_adapter = get_vector_store_adapter()
deployment_event_store = get_deployment_event_store()
registry_adapter = get_registry_adapter()
eval_result_adapter = get_eval_result_adapter()

EMBEDDING_MODEL: Final[str] = "voyage-3"


class RagIngestRequest(BaseModel):
    collection: str
    source_paths: list[str]
    chunk_size: int = 800
    chunk_overlap: int = 100


class RagIngestResponse(BaseModel):
    collection: str
    index_version: str
    chunks_ingested: int


class RagEvalCase(BaseModel):
    question: str


class RagEvaluateRequest(BaseModel):
    collection: str
    index_version: str
    eval_cases: list[RagEvalCase]
    top_k: int = 5
    # Overridable so RAG eval isn't locked to Claude.
    model: str = "claude-sonnet-5"


class RagEvaluateResponse(BaseModel):
    passed: bool
    pass_rate: float
    results: list[dict[str, object]]
    # Answer-generation calls only; total_cost_usd is None with no cost entry.
    total_tokens: int
    total_cost_usd: float | None


class RagActivateRequest(BaseModel):
    collection: str
    index_version: str
    # See ActivatePromptRequest.environment (routers/prompts.py) — same
    # convention, mirrored here for the RAG-index half of the lifecycle.
    environment: str = DEFAULT_ENVIRONMENT
    # See ActivatePromptRequest.is_rollback (routers/prompts.py) — same
    # convention, mirrored here for the RAG-index half of the lifecycle.
    is_rollback: bool = False


class RagActivateResponse(BaseModel):
    collection: str
    environment: str
    active_version: str


class RagActiveVersionResponse(BaseModel):
    collection: str
    environment: str
    active_version: str | None


class RagCollectionNamesResponse(BaseModel):
    names: list[str]


class RagIndexVersionsResponse(BaseModel):
    versions: list[str]


class RagSourceListResponse(BaseModel):
    sources: list[str]


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    step = max(1, chunk_size - chunk_overlap)
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


# Repo root, derived from this file (services/orchestration-api/routers/rag.py
# -> parents[3]). In the Docker image the service is copied to /app, so this
# resolves to "/" there — DOCS_ROOT (set in the image) covers that case.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]


def _docs_roots() -> list[Path]:
    """Directories that may contain the repo's `docs/` folder, most specific
    first. `docs_root` (the repo root, not docs/ itself) wins when set; the
    __file__-derived repo root is the local-dev fallback."""
    roots: list[Path] = []
    if settings.docs_root:
        roots.append(Path(settings.docs_root))
    roots.append(_REPO_ROOT)
    return roots


def _resolve_source_path(source_path: str) -> Path:
    """Resolve a repo-relative source path against every location it can
    legitimately live in, instead of only the process CWD.

    `make run-orchestration-api` runs uvicorn with CWD=services/orchestration-api,
    so a template's "docs/architecture-overview.md" would otherwise resolve to
    services/orchestration-api/docs/... (missing) rather than the repo's own
    docs/. The Docker image copies the service to /app and docs/ to /app/docs,
    where CWD already works; DOCS_ROOT covers any other layout.
    """
    candidate = Path(source_path)
    if candidate.is_absolute():
        return candidate
    for root in [Path.cwd(), *_docs_roots()]:
        resolved = root / candidate
        if resolved.is_file():
            return resolved
    return Path.cwd() / candidate


# Declared before the "/{collection}" catch-all below — FastAPI matches
# routes in declaration order, or "/collections" would swallow it.
@router.get("/collections", response_model=RagCollectionNamesResponse)
def list_rag_collections(user: dict = Depends(get_current_user)) -> RagCollectionNamesResponse:
    return RagCollectionNamesResponse(names=registry_adapter.list_names("rag-index"))


@router.get("/sources", response_model=RagSourceListResponse)
def list_rag_sources(user: dict = Depends(get_current_user)) -> RagSourceListResponse:
    """Repo-relative paths of every ingestable doc under `docs/` — backs the
    Draft/Ingest template's source picker, so a Dev picks a real path instead
    of typing one that 400s at ingest. Paths are returned as `docs/<rel>` to
    match exactly what `_resolve_source_path` accepts."""
    seen: set[str] = set()
    sources: list[str] = []
    for root in _docs_roots():
        docs_dir = root / "docs"
        if not docs_dir.is_dir():
            continue
        for path in sorted(docs_dir.rglob("*.md")):
            source = f"docs/{path.relative_to(docs_dir).as_posix()}"
            if source not in seen:
                seen.add(source)
                sources.append(source)
    return RagSourceListResponse(sources=sources)


@router.get("/collections/{name}/versions", response_model=RagIndexVersionsResponse)
def list_rag_collection_versions(
    name: str, user: dict = Depends(get_current_user)
) -> RagIndexVersionsResponse:
    versions = registry_adapter.list_versions("rag-index", name)
    return RagIndexVersionsResponse(versions=sorted(versions, key=int))


@router.get("/{collection}", response_model=RagActiveVersionResponse)
def get_rag_active_version(
    collection: str, environment: str = DEFAULT_ENVIRONMENT, user: dict = Depends(get_current_user)
) -> RagActiveVersionResponse:
    # Mirrors routers/prompts.py's list_prompts()/get_prompt() read pattern —
    # added for agents/mcp-servers/ai-observability-server's
    # get_active_rag_version tool, which had no endpoint to call before.
    # `environment` defaults to DEFAULT_ENVIRONMENT — same backward-compat
    # reasoning as get_prompt_active_version.
    return RagActiveVersionResponse(
        collection=collection,
        environment=environment,
        active_version=registry_adapter.get_active_version("rag-index", collection, environment),
    )


@router.post("/ingest", response_model=RagIngestResponse)
def rag_ingest(
    request: RagIngestRequest, user: dict = Depends(get_current_user)
) -> RagIngestResponse:
    chunks: list[str] = []
    sources: list[str] = []
    for source_path in request.source_paths:
        # Repo-relative path resolved against the repo root / DOCS_ROOT — 400
        # for a typo or an unmounted doc instead of a raw FileNotFoundError 500.
        resolved = _resolve_source_path(source_path)
        if not resolved.is_file():
            raise HTTPException(400, f"source file not found: {source_path}")
        text = resolved.read_text()
        for chunk in _chunk_text(text, request.chunk_size, request.chunk_overlap):
            chunks.append(chunk)
            sources.append(source_path)

    vectors = llm_gateway_adapter.embed(EMBEDDING_MODEL, chunks)
    vector_store_adapter.ensure_collection(
        vector_size=len(vectors[0]), collection=request.collection
    )
    # Register the version BEFORE upserting so every point can carry its
    # `index_version` — that payload field is what makes the version a real
    # retrieval boundary (search filters on it), not just a counter.
    index_version = registry_adapter.register_version(
        "rag-index",
        request.collection,
        {"chunks_ingested": len(chunks), "source_paths": request.source_paths},
    )
    ids = [str(uuid.uuid4()) for _ in chunks]
    payloads = [
        {"text": chunk, "source": source} for chunk, source in zip(chunks, sources, strict=True)
    ]
    vector_store_adapter.upsert(
        ids, vectors, payloads, collection=request.collection, index_version=index_version
    )

    # Attribute the embedding spend to the index version's build stage. The
    # embed() call returns no cost header, so this is priced from the real
    # chunk count at the reference embedding rate.
    record_cost_event(
        stage="build",
        artifact_kind="rag-index",
        artifact_id=request.collection,
        version=index_version,
        environment=DEFAULT_ENVIRONMENT,
        cost_usd=len(chunks) / 1000 * EMBED_USD_PER_1K_CHUNK,
        quantity=float(len(chunks)),
        unit="chunk",
        unit_price=EMBED_USD_PER_1K_CHUNK,
        source="litellm",
        run_id=f"ingest-{request.collection}-{index_version}",
    )

    return RagIngestResponse(
        collection=request.collection, index_version=index_version, chunks_ingested=len(chunks)
    )


@router.post("/evaluate", response_model=RagEvaluateResponse)
def rag_evaluate(
    request: RagEvaluateRequest, user: dict = Depends(get_current_user)
) -> RagEvaluateResponse:
    # The UI shows versions as "v1" while the store keys them as "1" — accept
    # either so a typed "v1" still scopes retrieval to the right index.
    request.index_version = normalize_version(request.index_version)
    results: list[dict[str, object]] = []
    total_tokens = 0
    total_cost_usd = 0.0
    cost_known = True
    passed_count = 0
    for eval_case in request.eval_cases:
        query_vector = llm_gateway_adapter.embed(EMBEDDING_MODEL, [eval_case.question])[0]
        hits = vector_store_adapter.search(
            query_vector,
            top_k=request.top_k,
            collection=request.collection,
            index_version=request.index_version,
        )
        context = "\n\n".join(str(hit["payload"]["text"]) for hit in hits)
        system_prompt = f"Answer using only this context:\n\n{context}"
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
        judge_result = judge_response(eval_case.question, answer, context)
        gate_result = evaluate_gate(judge_result)
        results.append(
            {"question": eval_case.question, "answer": answer, "passed": gate_result["passed"]}
        )
        if gate_result["passed"]:
            passed_count += 1

        # Persist judge result for MTTR calculation
        eval_result_adapter.log_judge_result(
            kind="rag-index",
            name=request.collection,
            version=request.index_version,
            judge_result=judge_result,
            passed=gate_result["passed"],
        )

    pass_rate = passed_count / len(results) if results else 0.0
    overall_passed = pass_rate >= 0.8

    # Attribute the real judge spend to the index version's gate stage.
    record_cost_event(
        stage="gate",
        artifact_kind="rag-index",
        artifact_id=request.collection,
        version=request.index_version,
        environment=DEFAULT_ENVIRONMENT,
        cost_usd=total_cost_usd if cost_known else 0.0,
        quantity=float(total_tokens),
        unit="token",
        source="litellm",
        run_id=f"eval-rag-{request.collection}-{request.index_version}",
    )

    # Emit DORA gate evaluation metric (LLMOps track, rag-index)
    GATE_EVALUATIONS.labels(
        track="llmops",
        subject_type="rag-index",
        subject_id=f"{request.collection}:{request.index_version}",
        passed=str(overall_passed).lower(),
    ).inc()

    return RagEvaluateResponse(
        passed=overall_passed,
        pass_rate=pass_rate,
        results=results,
        total_tokens=total_tokens,
        total_cost_usd=total_cost_usd if cost_known else None,
    )


@router.post("/activate", response_model=RagActivateResponse)
def rag_activate(
    request: RagActivateRequest, user: dict = Depends(get_current_user)
) -> RagActivateResponse:
    request.index_version = normalize_version(request.index_version)
    registry_adapter.set_active_version(
        "rag-index", request.collection, request.index_version, request.environment
    )

    event_type = "rollback" if request.is_rollback else "deploy"

    # Emit DORA deployment event (LLMOps track)
    DEPLOYMENT_EVENTS.labels(
        track="llmops",
        subject_type="rag-index",
        subject_id=request.collection,
        event_type=event_type,
    ).inc()
    now = datetime.now().isoformat()
    deployment_event_store.record_event(
        name=f"rag-{event_type}-{request.collection}-{now}",
        change_type="rag_index",
        project_name=request.collection,
        component_name="rag-index",
        # Reflects the environment actually activated — see
        # routers/prompts.py's activate_prompt for why this was a bug
        # before (hardcoded "development" regardless of what happened).
        environment_name=request.environment,
        outcome="success",
        started_at=now,
        finished_at=now,
    )

    # MTTR: find last judge failure for this rag-index and calculate recovery time
    last_failure = eval_result_adapter.get_last_failure_at("rag-index", request.collection)
    if last_failure is not None:
        recovery_seconds = (datetime.now() - last_failure).total_seconds()
        INCIDENT_RECOVERY.labels(
            track="llmops",
            subject_type="rag-index",
            subject_id=request.collection,
        ).observe(recovery_seconds)

    return RagActivateResponse(
        collection=request.collection,
        environment=request.environment,
        active_version=request.index_version,
    )
