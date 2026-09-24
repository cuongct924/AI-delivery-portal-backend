"""Eval Set Registry API — named, versioned sets of eval questions reused
across prompt/RAG-index evaluation runs (routers/prompts.py, routers/rag.py),
so two versions of the same artifact are graded against the same benchmark
instead of whatever questions happened to be typed into that run's form.

Backed by the same QdrantVersionRegistryAdapter as routers/rag.py's
"rag-index" kind, under kind="eval-set" — it was already designed and
tested as a multi-kind store (see that adapter's module docstring).
"""

from typing import Final, cast

from auth.thunder import get_current_user
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from adapters.ai_platform.interfaces import normalize_version
from adapters.factory import get_registry_adapter

router = APIRouter(prefix="/eval-sets", tags=["eval-sets"])

registry_adapter = get_registry_adapter()

_KIND: Final[str] = "eval-set"


class EvalSetNamesResponse(BaseModel):
    names: list[str]


class EvalSetVersionsResponse(BaseModel):
    versions: list[str]


class DraftEvalSetRequest(BaseModel):
    name: str
    questions: list[str]


class DraftEvalSetResponse(BaseModel):
    name: str
    version: str
    questions: list[str]


class EvalSetVersionResponse(BaseModel):
    name: str
    version: str
    questions: list[str]


# Declared before the "/{name}/{version}" catch-all below — FastAPI matches
# routes in declaration order, same convention as routers/rag.py's
# "/collections" vs "/{collection}".
@router.get("", response_model=EvalSetNamesResponse)
def list_eval_set_names(user: dict = Depends(get_current_user)) -> EvalSetNamesResponse:
    return EvalSetNamesResponse(names=registry_adapter.list_names(_KIND))


@router.get("/{name}/versions", response_model=EvalSetVersionsResponse)
def list_eval_set_versions(
    name: str, user: dict = Depends(get_current_user)
) -> EvalSetVersionsResponse:
    versions = registry_adapter.list_versions(_KIND, name)
    return EvalSetVersionsResponse(versions=sorted(versions, key=int))


def _not_found_detail(name: str) -> str:
    """Actionable 404 body for a name that was never drafted — the caller
    (llm-evaluate-activate's fetch-eval-set step) only has a free-text name
    to go on, so listing what *does* exist turns "not found" into "here's
    the name you meant" without a second round-trip."""
    available = registry_adapter.list_names(_KIND)
    if not available:
        return (
            f"Eval set not found: {name!r}. No eval sets registered yet — draft one "
            "via the 'Draft Prompt / Ingest RAG Data / Draft Eval Set' template "
            "(artifactKind=eval-set) first."
        )
    return f"Eval set not found: {name!r}. Available eval sets: {sorted(available)}"


@router.get("/{name}/latest", response_model=EvalSetVersionResponse)
def get_latest_eval_set_version(
    name: str, user: dict = Depends(get_current_user)
) -> EvalSetVersionResponse:
    """The most recently registered version's questions — what
    llm-evaluate-activate's fetch-eval-set step pulls before running the
    Evaluate Gate, so picking an eval set by name always uses its newest
    revision without the caller tracking version numbers itself."""
    versions = registry_adapter.list_versions(_KIND, name)
    if not versions:
        raise HTTPException(404, _not_found_detail(name))
    latest_version = max(versions, key=int)
    metadata = registry_adapter.get_version(_KIND, name, latest_version)
    return EvalSetVersionResponse(
        name=name, version=latest_version, questions=cast(list[str], metadata["questions"])
    )


@router.post("", response_model=DraftEvalSetResponse)
def draft_eval_set(
    request: DraftEvalSetRequest, user: dict = Depends(get_current_user)
) -> DraftEvalSetResponse:
    version = registry_adapter.register_version(
        _KIND, request.name, {"questions": request.questions}
    )
    return DraftEvalSetResponse(name=request.name, version=version, questions=request.questions)


@router.get("/{name}/{version}", response_model=EvalSetVersionResponse)
def get_eval_set_version(
    name: str, version: str, user: dict = Depends(get_current_user)
) -> EvalSetVersionResponse:
    version = normalize_version(version)
    try:
        metadata = registry_adapter.get_version(_KIND, name, version)
    except ValueError as exc:
        raise HTTPException(404, f"Eval set not found: {name}/{version}") from exc
    return EvalSetVersionResponse(
        name=name, version=version, questions=cast(list[str], metadata["questions"])
    )
