"""Adapter for AI Notebook (JupyterHub) — self-hosted, provisions a per-user
Jupyter server via a KubeSpawner profile (environment/CPU/RAM/GPU/storage).

Mirrors the Viettel AI Notebooks service shape: independent notebooks with
CPU/GPU/RAM/Storage, on/off lifecycle, idle auto-shutdown, and a persistent
per-user working environment (the spawner's storage volume survives
stop/start, so `start_notebook` resumes rather than recreates).

Note: needs a real JupyterHub deployment (KubeSpawner + profile_list
matching the environment/gpu_type choices) before this connects for real —
same infra-phase caveat as kserve_adapter.py. JupyterHub's user API doesn't
echo the spawner profile back, so the resource fields are cached in-process
on create; a restart loses them until the next create.
"""

import os
import uuid
from datetime import datetime

import httpx

from adapters.ai_platform.interfaces import (
    INotebookAdapter,
    NotebookDeletion,
    NotebookSpec,
    NotebookStatus,
)

# Must match the `slug` on the single profile in infra/jupyterhub/values.yaml
# — KubeSpawner otherwise derives it from display_name.
_PROFILE_SLUG = "ai-notebook"


class JupyterHubAdapter(INotebookAdapter):
    def __init__(self, base_url: str | None = None, token: str | None = None):
        # Hub API port, not the proxy's 8000 — and never the orchestration
        # API's own localhost:8000, which the old default collided with
        # (the API ended up calling itself and 404ing).
        self.base_url = base_url or os.getenv("JUPYTERHUB_URL", "http://localhost:8081")
        self.token = token or os.getenv("JUPYTERHUB_API_TOKEN", "")
        # Where a browser reaches the proxy (the notebook's own URL) — the
        # hub API returns a relative `/user/<name>/` path, which is useless
        # as a link from the Portal. Port 8888, not 8080: a forward on 8080
        # shadows `thunder.openchoreo.localhost` (resolves to 127.0.0.1) and
        # breaks Backstage sign-in.
        self.public_url = os.getenv("JUPYTERHUB_PUBLIC_URL", "http://localhost:8888").rstrip("/")
        # Only send the header when a token exists — an empty one produced
        # "Illegal header value b'token '" and 500'd every call.
        self._headers = {"Authorization": f"token {self.token}"} if self.token else {}
        # notebook_id -> the profile it was spawned with (see module docstring).
        self._specs: dict[str, NotebookSpec] = {}

    def _ensure_user(self, notebook_id: str) -> None:
        """JupyterHub's spawn endpoint 404s for a user that doesn't exist
        yet — create it first. 409 (already exists) is the normal re-spawn
        case, so it's swallowed."""
        response = httpx.post(
            f"{self.base_url}/hub/api/users",
            json={"usernames": [notebook_id]},
            headers=self._headers,
            timeout=30,
        )
        if response.status_code not in (201, 409):
            response.raise_for_status()

    def _spawn(self, notebook_id: str) -> None:
        spec = self._specs[notebook_id]
        self._ensure_user(notebook_id)
        # KubeSpawner reads user_options as FLAT keys — `profile` plus one
        # key per profile_option — not a nested `profile_options` dict (that
        # shape is silently ignored, leaving every option on its default).
        # Values are choice KEYS (strings), so stringify the numeric ones.
        response = httpx.post(
            f"{self.base_url}/hub/api/users/{notebook_id}/server",
            json={
                "profile": _PROFILE_SLUG,
                "environment": spec["environment"],
                "cpu_cores": str(spec["cpu_cores"]),
                "ram_gb": str(spec["ram_gb"]),
                "gpu_type": spec["gpu_type"] or "none",
                "gpu_count": str(spec["gpu_count"]),
                "storage_gb": str(spec["storage_gb"]),
                "idle_timeout_minutes": str(spec["idle_timeout_minutes"]),
            },
            headers=self._headers,
            timeout=30,
        )
        response.raise_for_status()

    def create_notebook(
        self,
        environment: str,
        ram_gb: int,
        gpu_type: str | None = None,
        *,
        cpu_cores: int = 2,
        gpu_count: int = 1,
        storage_gb: int = 20,
        idle_timeout_minutes: int = 60,
    ) -> NotebookStatus:
        notebook_id = f"nb-{uuid.uuid4().hex[:8]}"
        self._specs[notebook_id] = {
            "environment": environment,
            "cpu_cores": cpu_cores,
            "ram_gb": ram_gb,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count if gpu_type else 0,
            "storage_gb": storage_gb,
            "idle_timeout_minutes": idle_timeout_minutes,
            "created_at": datetime.utcnow().isoformat(),
        }
        self._spawn(notebook_id)
        return self.get_notebook_status(notebook_id)

    def get_notebook_status(self, notebook_id: str) -> NotebookStatus:
        response = httpx.get(
            f"{self.base_url}/hub/api/users/{notebook_id}", headers=self._headers, timeout=10
        )
        # 404 -> ValueError so the router's _get_or_404 turns it into a clean
        # 404 instead of a 500 (same convention as MlflowAdapter).
        if response.status_code == 404:
            raise ValueError(f"notebook {notebook_id} not found")
        response.raise_for_status()
        data = response.json()
        spec = self._specs.get(notebook_id, {})
        # The URL is deterministic from the user name, so return it even
        # while the server is still starting — create_notebook() returns
        # right after spawn, and a None url there left the Portal's "Open
        # the notebook" link empty. `active` still reflects real readiness.
        return {
            "notebook_id": notebook_id,
            "url": f"{self.public_url}/user/{notebook_id}/",
            "active": data.get("server") is not None,
            "environment": str(spec.get("environment", "")),
            "cpu_cores": int(spec.get("cpu_cores", 0)),
            "ram_gb": int(spec.get("ram_gb", 0)),
            "gpu_type": spec.get("gpu_type"),  # type: ignore[typeddict-item]
            "gpu_count": int(spec.get("gpu_count", 0)),
            "storage_gb": int(spec.get("storage_gb", 0)),
            "idle_timeout_minutes": int(spec.get("idle_timeout_minutes", 0)),
            "created_at": str(spec.get("created_at", "")),
            "last_activity_at": data.get("last_activity"),
        }

    def list_notebooks(self) -> list[NotebookStatus]:
        response = httpx.get(f"{self.base_url}/hub/api/users", headers=self._headers, timeout=10)
        response.raise_for_status()
        # Only the users this adapter spawned (nb- prefix) are notebooks.
        return [
            self.get_notebook_status(user["name"])
            for user in response.json()
            if str(user.get("name", "")).startswith("nb-")
        ]

    def start_notebook(self, notebook_id: str) -> NotebookStatus:
        self._spawn(notebook_id)
        return self.get_notebook_status(notebook_id)

    def stop_notebook(self, notebook_id: str) -> NotebookStatus:
        response = httpx.delete(
            f"{self.base_url}/hub/api/users/{notebook_id}/server",
            headers=self._headers,
            timeout=30,
        )
        response.raise_for_status()
        return self.get_notebook_status(notebook_id)

    def delete_notebook(self, notebook_id: str) -> NotebookDeletion:
        response = httpx.delete(
            f"{self.base_url}/hub/api/users/{notebook_id}",
            headers=self._headers,
            timeout=30,
        )
        response.raise_for_status()
        self._specs.pop(notebook_id, None)
        return {"notebook_id": notebook_id, "deleted": True}
