"""Read-only lookup of the mlops/llmops Golden Path Scaffolder templates —
backs the golden-path-guide-server MCP tools (list/describe "how do I run
Golden Path X"). Reads straight from the Backstage Catalog
(catalog_client.py) so the answer always matches the live template.yaml,
never a separately-maintained copy.
"""

import json
from typing import Any, cast

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


def _render_condition(cond: dict[str, Any]) -> str | None:
    """Render a schema `if` clause as `field=value[ and field2=value2]`.
    Returns None when the clause uses `not`/`anyOf`/`oneOf` — safely
    rendering those in prose (not just AND-of-equalities) risks silently
    misrepresenting the condition, which is worse than not compressing it.
    Measured against the real train-track-register schema: 47 of its 48
    `allOf` branches are plain AND-of-equalities; only 1 needs this
    fallback."""
    if "not" in cond or "anyOf" in cond or "oneOf" in cond:
        return None
    parts: list[str] = []
    for field, spec in (cond.get("properties") or {}).items():
        if "const" in spec:
            parts.append(f"{field}={spec['const']}")
        elif "enum" in spec:
            vals = spec["enum"]
            parts.append(f"{field} in [{','.join(str(v) for v in vals)}]")
        else:
            return None
    return " and ".join(parts) if parts else None


def _render_field(name: str, schema: dict[str, Any], required: set[str]) -> str:
    title = schema.get("title", "")
    label = f"{name}*" if name in required else name
    if title:
        label += f" ({title})"
    one_of = schema.get("oneOf")
    if one_of and all({"const", "title"} >= set(o.keys()) for o in one_of):
        opts = " | ".join(f"{o['const']}={o.get('title', o['const'])}" for o in one_of)
        return f"- {label}: {opts}"
    enum = schema.get("enum")
    if enum:
        return f"- {label}: enum[{','.join(str(e) for e in enum)}]"
    if schema.get("const") is not None:
        return f"- {label}: const={schema['const']}"
    typ = schema.get("type", "string")
    # First sentence only, and drop the "Default: X" clause the mlops
    # persona prompt already tells the agent to ignore for business-
    # defining fields — both are pure token cost with no signal left to
    # extract once that rule exists.
    desc = (schema.get("description") or "").split("\n")[0]
    desc = desc.split(". Default:")[0].split(", Default:")[0].strip()[:100]
    return f"- {label}: {typ}" + (f" — {desc}" if desc else "")


def summarize_golden_path_schema(name: str) -> str | None:
    """Condensed, human-readable rendering of a template's schema for
    embedding in the agent's context — same information the raw JSON
    Schema carries (fields/types/options/required-ness/branching) at
    roughly half the token cost (measured: 58% smaller on
    train-track-register, the largest of the 6 templates). `readOnly`
    fields (costEstimate/securityScan — see template.yaml's own comment:
    "Its own value is never read") are dropped entirely; they're pure
    plumbing for the Scaffolder wizard's panels, irrelevant to filling a
    draft. Returns None when the template doesn't exist, same as
    get_golden_path_schema."""
    groups = get_golden_path_schema(name)
    if groups is None:
        return None
    lines: list[str] = []
    for raw_group in groups:
        group = cast(dict[str, Any], raw_group)
        lines.append(f"## {group.get('title', '')}")
        required = set(group.get("required") or [])
        for fname, fschema in (group.get("properties") or {}).items():
            if fschema.get("readOnly"):
                continue
            lines.append(_render_field(fname, fschema, required))
        for branch in group.get("allOf") or []:
            then = branch.get("then") or {}
            then_required = set(then.get("required") or [])
            then_props = then.get("properties") or {}
            cond_text = _render_condition(branch.get("if") or {})
            if cond_text is None:
                # Too complex to safely render as prose — embed this one
                # branch's raw JSON rather than guess.
                lines.append(f"[condition]: {json.dumps(branch, separators=(',', ':'))}")
                continue
            if not then_props and not then_required:
                continue
            lines.append(f"when {cond_text}:")
            for fname, fschema in then_props.items():
                if fschema.get("readOnly"):
                    continue
                lines.append("  " + _render_field(fname, fschema, then_required))
            for fname in then_required:
                if fname not in then_props:
                    lines.append(f"  - {fname}*")
    return "\n".join(lines)


@router.get("/{name}/schema/summary")
def get_golden_path_schema_summary_endpoint(
    name: str, user: dict = Depends(get_current_user)
) -> dict[str, str]:
    """Condensed form of `/schema` for the chat agent's context — see
    summarize_golden_path_schema's docstring. `/draft` still validates
    against the full schema; only what gets embedded in prompts changes."""
    del user
    summary = summarize_golden_path_schema(name)
    if summary is None:
        raise HTTPException(404, f"no golden path template named {name!r}")
    return {"summary": summary}


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


def validate_golden_path_draft(name: str, values: dict[str, Any]) -> GoldenPathDraftResponse | None:
    """Validate a set of form values against a template's live schema. Pure —
    no side effect, no submit. `missing` lists required fields the proposal
    left out (branch-aware, derived from the validator's own `required`
    errors) so a caller can fill them on the next turn. Backs the `/draft`
    endpoint the golden-path-guide-server MCP tool calls. (Session-restore
    revalidation used to be a second caller here too — moved to the
    portal-assistant-backend Node plugin's own DraftService, frontend repo,
    once it became the draft's sole owner.) Returns None when the template
    doesn't exist."""
    groups = get_golden_path_schema(name)
    if groups is None:
        return None

    form_data = dict(values)
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


@router.post("/{name}/draft", response_model=GoldenPathDraftResponse)
def propose_golden_path_draft(
    name: str, request: GoldenPathDraftRequest, user: dict = Depends(get_current_user)
) -> GoldenPathDraftResponse:
    """Validate an agent-proposed set of form values against the template's
    live schema. Pure — no side effect, no submit. `missing` lists required
    fields the proposal left out (branch-aware, derived from the validator's
    own `required` errors) so the agent can fill them on the next turn."""
    del user
    result = validate_golden_path_draft(name, request.values)
    if result is None:
        raise HTTPException(404, f"no golden path template named {name!r}")
    return result


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
