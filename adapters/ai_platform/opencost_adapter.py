"""OpenCost-backed cost allocation.

OpenCost (CNCF, Apache-2.0) is the real source of Kubernetes cost allocation —
the observer cost API's synthetic baseline is a stand-in for it. When
`OPENCOST_URL` is set, the observer reads OpenCost's `/allocation` API; otherwise
it falls back to the deterministic synthetic baseline so the page renders
without an observability plane.

Kept as plain functions (not an Adapter class) because it has no state and no
mock/real split of its own — the caller decides whether to use it.
"""

import logging
import os
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def opencost_url() -> str | None:
    return os.getenv("OPENCOST_URL") or None


def fetch_allocation(namespace: str, start: datetime, end: datetime) -> list[dict[str, Any]] | None:
    """Allocation rows aggregated by namespace+controller, or None when OpenCost
    isn't configured or the call fails (the caller falls back to synthetic)."""
    url = opencost_url()
    if not url:
        return None
    try:
        response = httpx.get(
            f"{url.rstrip('/')}/allocation",
            params={
                "window": f"{start.isoformat()},{end.isoformat()}",
                "aggregate": "namespace,controller",
                "accumulate": "true",
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json().get("data") or []
    except Exception as exc:  # noqa: BLE001 — fall back to synthetic
        logger.warning("OpenCost allocation fetch failed: %s", exc)
        return None


def to_cost_items(allocation: list[dict[str, Any]], environment: str) -> list[dict[str, Any]]:
    """Map OpenCost allocation rows to the observer's CostItem shape. Each row is
    a dict keyed by the aggregation value (e.g. "default/serving") whose value is
    the allocation object."""
    items: list[dict[str, Any]] = []
    for row in allocation:
        for key, alloc in row.items():
            if not isinstance(alloc, dict):
                continue
            props = alloc.get("properties") or {}
            window = alloc.get("window") or {}
            controller = str(props.get("controller") or key)
            ns = str(props.get("namespace") or "default")
            items.append(
                {
                    "component": controller,
                    "startTime": str(window.get("start") or ""),
                    "endTime": str(window.get("end") or ""),
                    "environment": environment,
                    "project": ns,
                    "namespace": ns,
                    "cpuCost": float(alloc.get("cpuCost") or 0.0),
                    "memoryCost": float(alloc.get("ramCost") or 0.0),
                    "efficiency": float(alloc.get("efficiency") or 0.0),
                    "artifact": controller,
                }
            )
    return items
