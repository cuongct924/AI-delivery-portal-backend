"""Read-only lookup of the mlops/llmops Golden Path Scaffolder templates —
backs the golden-path-guide-server MCP tools (list/describe "how do I run
Golden Path X"). Reads straight from the Backstage Catalog
(catalog_client.py) so the answer always matches the live template.yaml,
never a separately-maintained copy.
"""

from auth.thunder import get_current_user
from catalog_client import get_golden_path_template, list_golden_path_templates
from fastapi import APIRouter, Depends, HTTPException
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
