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


class JupyterHubAdapter(INotebookAdapter):
    def __init__(self, base_url: str | None = None, token: str | None = None):
        self.base_url = base_url or os.getenv("JUPYTERHUB_URL", "http://localhost:8000")
        self.token = token or os.getenv("JUPYTERHUB_API_TOKEN", "")
        self._headers = {"Authorization": f"token {self.token}"}
        # notebook_id -> the profile it was spawned with (see module docstring).
        self._specs: dict[str, NotebookSpec] = {}

    def _spawn(self, notebook_id: str) -> None:
        spec = self._specs[notebook_id]
        response = httpx.post(
            f"{self.base_url}/hub/api/users/{notebook_id}/server",
            json={
                "profile_options": {
                    "environment": spec["environment"],
                    "cpu_cores": spec["cpu_cores"],
                    "ram_gb": spec["ram_gb"],
                    "gpu_type": spec["gpu_type"] or "none",
                    "gpu_count": spec["gpu_count"],
                    "storage_gb": spec["storage_gb"],
                    "idle_timeout_minutes": spec["idle_timeout_minutes"],
                }
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
        response.raise_for_status()
        data = response.json()
        spec = self._specs.get(notebook_id, {})
        return {
            "notebook_id": notebook_id,
            "url": data.get("server"),
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
