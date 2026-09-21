"""Adapter for AI Notebook (JupyterHub) — self-hosted, provisions a per-user
Jupyter server via a KubeSpawner profile (environment/RAM/GPU).

Note: needs a real JupyterHub deployment (KubeSpawner + profile_list
matching the environment/gpu_type choices) before this connects for real —
same infra-phase caveat as kserve_adapter.py.
"""

import os
import uuid

import httpx

from adapters.ai_platform.interfaces import INotebookAdapter, NotebookDeletion, NotebookStatus


class JupyterHubAdapter(INotebookAdapter):
    def __init__(self, base_url: str | None = None, token: str | None = None):
        self.base_url = base_url or os.getenv("JUPYTERHUB_URL", "http://localhost:8000")
        self.token = token or os.getenv("JUPYTERHUB_API_TOKEN", "")
        self._headers = {"Authorization": f"token {self.token}"}

    def create_notebook(
        self, environment: str, ram_gb: int, gpu_type: str | None = None
    ) -> NotebookStatus:
        notebook_id = f"nb-{uuid.uuid4().hex[:8]}"
        response = httpx.post(
            f"{self.base_url}/hub/api/users/{notebook_id}/server",
            json={
                "profile_options": {
                    "environment": environment,
                    "ram_gb": ram_gb,
                    "gpu_type": gpu_type or "none",
                }
            },
            headers=self._headers,
            timeout=30,
        )
        response.raise_for_status()
        return self.get_notebook_status(notebook_id)

    def get_notebook_status(self, notebook_id: str) -> NotebookStatus:
        response = httpx.get(
            f"{self.base_url}/hub/api/users/{notebook_id}", headers=self._headers, timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return {
            "notebook_id": notebook_id,
            "url": data.get("server"),
            "active": data.get("server") is not None,
        }

    def delete_notebook(self, notebook_id: str) -> NotebookDeletion:
        response = httpx.delete(
            f"{self.base_url}/hub/api/users/{notebook_id}/server",
            headers=self._headers,
            timeout=30,
        )
        response.raise_for_status()
        return {"notebook_id": notebook_id, "deleted": True}
