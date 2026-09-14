"""Reads Backstage Catalog entities directly — MCP server discovery
(`API, type: mcp` entities) and Golden Path template lookup
(`kind: Template` entities), so neither needs a hardcoded local copy of
data Backstage already owns.

Requires a static service token (`settings.backstage_service_token`,
matching app-config.yaml's `backend.auth.externalAccess`) — Backstage
rejects unauthenticated Catalog reads.
"""

import logging
from typing import TypedDict

import httpx
from core.config import settings

logger = logging.getLogger("orchestration_api.catalog_client")

# Golden Path templates are tagged with one of these in template.yaml —
# everything else in the Catalog (ClusterComponentType/etc. scaffolding
# templates) is out of scope for golden-path lookup.
_GOLDEN_PATH_TAGS = {"mlops", "llmops"}


class McpServerInfo(TypedDict):
    name: str
    endpoint: str
    transport: str


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


class GoldenPathDetail(GoldenPathSummary):
    parameters: list[GoldenPathParameter]
    steps: list[GoldenPathStep]


def _auth_headers() -> dict[str, str]:
    if settings.backstage_service_token:
        return {"Authorization": f"Bearer {settings.backstage_service_token}"}
    return {}


def discover_mcp_servers() -> list[McpServerInfo]:
    """Return connection info for every `API, type: mcp` Catalog entity.
    Never raises — an unreachable Catalog just yields an empty list."""
    try:
        response = httpx.get(
            f"{settings.backstage_base_url}/api/catalog/entities",
            params={"filter": "kind=api,spec.type=mcp"},
            headers=_auth_headers(),
            timeout=10,
        )
        response.raise_for_status()
        entities = response.json()
    except Exception:
        logger.warning("Backstage Catalog unreachable — MCP discovery skipped", exc_info=True)
        return []

    servers: list[McpServerInfo] = []
    for entity in entities:
        metadata = entity.get("metadata", {})
        annotations = metadata.get("annotations", {})
        name = metadata.get("name", "unknown")
        endpoint = annotations.get("mcp/endpoint")
        transport = annotations.get("mcp/transport")
        if not endpoint or not transport:
            logger.warning(
                "Catalog entity %s missing mcp/endpoint or mcp/transport annotation, skipping",
                name,
            )
            continue
        servers.append({"name": name, "endpoint": endpoint, "transport": transport})

    return servers


def list_golden_path_templates() -> list[GoldenPathSummary]:
    """Every Scaffolder Template Catalog entity tagged mlops/llmops — the
    Golden Path templates devs run from Backstage's Create page. Reads
    live from the Catalog rather than a hardcoded list, so a newly added
    template shows up with no code change here. Never raises — an
    unreachable Catalog just yields an empty list."""
    try:
        response = httpx.get(
            f"{settings.backstage_base_url}/api/catalog/entities",
            params={"filter": "kind=template"},
            headers=_auth_headers(),
            timeout=10,
        )
        response.raise_for_status()
        entities = response.json()
    except Exception:
        logger.warning("Backstage Catalog unreachable — golden path listing skipped", exc_info=True)
        return []

    summaries: list[GoldenPathSummary] = []
    for entity in entities:
        metadata = entity.get("metadata", {})
        tags = metadata.get("tags", [])
        if not _GOLDEN_PATH_TAGS.intersection(tags):
            continue
        name = metadata.get("name", "unknown")
        summaries.append(
            {
                "name": name,
                "title": metadata.get("title", name),
                "description": metadata.get("description", "").strip(),
                "tags": tags,
            }
        )
    return summaries


def get_golden_path_template(name: str) -> GoldenPathDetail | None:
    """One Golden Path template's full parameter/step spec, straight from
    its live Catalog entity — always in sync with the actual
    template.yaml, no separate copy to keep updated. Returns None when the
    entity doesn't exist, isn't tagged mlops/llmops, or the Catalog is
    unreachable."""
    try:
        response = httpx.get(
            f"{settings.backstage_base_url}/api/catalog/entities/by-name/template/default/{name}",
            headers=_auth_headers(),
            timeout=10,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        entity = response.json()
    except Exception:
        logger.warning("Backstage Catalog unreachable — golden path detail skipped", exc_info=True)
        return None

    metadata = entity.get("metadata", {})
    tags = metadata.get("tags", [])
    if not _GOLDEN_PATH_TAGS.intersection(tags):
        return None

    parameters: list[GoldenPathParameter] = []
    for group in entity.get("spec", {}).get("parameters", []):
        required = set(group.get("required", []))
        for prop_name, prop in group.get("properties", {}).items():
            parameters.append(
                {
                    "name": prop_name,
                    "title": prop.get("title", prop_name),
                    "description": prop.get("description", "").strip(),
                    "required": prop_name in required,
                }
            )

    steps: list[GoldenPathStep] = [
        {
            "name": step.get("name", step.get("id", "unknown")),
            "action": step.get("action", "unknown"),
        }
        for step in entity.get("spec", {}).get("steps", [])
    ]

    return {
        "name": metadata.get("name", name),
        "title": metadata.get("title", name),
        "description": metadata.get("description", "").strip(),
        "tags": tags,
        "parameters": parameters,
        "steps": steps,
    }
