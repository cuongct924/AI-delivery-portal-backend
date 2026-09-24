"""Read-only lookup of the mlops/llmops Golden Path Scaffolder templates —
backs the golden-path-guide-server MCP tools (list/describe "how do I run
Golden Path X"). Reads straight from the Backstage Catalog
(catalog_client.py) so the answer always matches the live template.yaml,
never a separately-maintained copy.
"""

from typing import Any

from auth.thunder import get_current_user
from catalog_client import (
    get_golden_path_schema,
    get_golden_path_template,
    list_golden_path_templates,
)
from fastapi import APIRouter, Depends, HTTPException
from jsonschema import Draft7Validator
from pydantic import BaseModel

router = APIRouter(prefix="/golden-paths", tags=["golden-paths"])


class GoldenPathSummaryResponse(BaseModel):
    name: str
    title: str
    description: str
    tags: list[str]


class GoldenPathParameterResponse(BaseModel):
    name: str
    title: str
    description: str
    required: bool


class GoldenPathStepResponse(BaseModel):
    name: str
    action: str


class GoldenPathDetailResponse(GoldenPathSummaryResponse):
    parameters: list[GoldenPathParameterResponse]
    steps: list[GoldenPathStepResponse]


@router.get("", response_model=list[GoldenPathSummaryResponse])
def list_golden_paths(
    user: dict = Depends(get_current_user),
) -> list[GoldenPathSummaryResponse]:
    del user
    return [GoldenPathSummaryResponse(**t) for t in list_golden_path_templates()]


@router.get("/{name}/schema")
def get_golden_path_schema_endpoint(
    name: str, user: dict = Depends(get_current_user)
) -> list[dict[str, object]]:
    """The template's raw `spec.parameters` JSON schema — what an agent
    needs to fill the form (types/enums/defaults/branching), unlike the
    flattened parameter list on the detail endpoint."""
    del user
    schema = get_golden_path_schema(name)
    if schema is None:
        raise HTTPException(404, f"no golden path template named {name!r}")
    return schema


class GoldenPathDraftRequest(BaseModel):
    values: dict[str, Any]


class GoldenPathDraftResponse(BaseModel):
    ok: bool
    form_data: dict[str, Any]
    missing: list[str]
    errors: list[str]


def _combined_schema(groups: list[dict[str, object]]) -> dict[str, Any]:
    """Merge a template's per-step parameter groups into one object schema
    so a single validator call covers the whole form. `allOf` branches are
    concatenated — their `if` conditions reference properties that are all
    present in the merged `properties`, so they still resolve."""
    combined: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
        "allOf": [],
    }
    for group in groups:
        combined["properties"].update(group.get("properties", {}) or {})
        combined["required"].extend(group.get("required", []) or [])
        combined["allOf"].extend(group.get("allOf", []) or [])
    return combined


@router.post("/{name}/draft", response_model=GoldenPathDraftResponse)
def propose_golden_path_draft(
    name: str, request: GoldenPathDraftRequest, user: dict = Depends(get_current_user)
) -> GoldenPathDraftResponse:
    """Validate an agent-proposed set of form values against the template's
    live schema. Pure — no side effect, no submit. `missing` lists required
    fields the proposal left out (branch-aware, derived from the validator's
    own `required` errors) so the agent can fill them on the next turn."""
    del user
    groups = get_golden_path_schema(name)
    if groups is None:
        raise HTTPException(404, f"no golden path template named {name!r}")

    form_data = dict(request.values)
    validator = Draft7Validator(_combined_schema(groups))
    errors = sorted(validator.iter_errors(form_data), key=lambda e: list(e.path))

    missing: list[str] = []
    for error in errors:
        if error.validator == "required":
            for prop in error.validator_value:
                if prop not in form_data and prop not in missing:
                    missing.append(prop)

    return GoldenPathDraftResponse(
        ok=not errors,
        form_data=form_data,
        missing=sorted(missing),
        errors=[error.message for error in errors],
    )


@router.get("/{name}", response_model=GoldenPathDetailResponse)
def get_golden_path(name: str, user: dict = Depends(get_current_user)) -> GoldenPathDetailResponse:
    del user
    template = get_golden_path_template(name)
    if template is None:
        raise HTTPException(404, f"no golden path template named {name!r}")
    return GoldenPathDetailResponse(
        name=template["name"],
        title=template["title"],
        description=template["description"],
        tags=template["tags"],
        parameters=[GoldenPathParameterResponse(**p) for p in template["parameters"]],
        steps=[GoldenPathStepResponse(**s) for s in template["steps"]],
    )
