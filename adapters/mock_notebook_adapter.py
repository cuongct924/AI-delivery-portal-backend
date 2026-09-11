"""Mock adapter for INotebookAdapter — stands in for the AI Notebook product
(JupyterHub, owned by a separate team, see docs/playbook-ai-delivery-portal.md's
AI Platform box). Not wired into any router yet (AI Notebook is an
independent Portal feature, not part of either Golden Path — see the
playbook's "no golden path #3" note), kept here so the adapter exists ahead
of that feature landing, same as JupyterHubAdapter itself.

Enable via `USE_MOCK_ADAPTERS=true` (adapters/factory.py).
"""

import uuid

from adapters.interfaces import INotebookAdapter, NotebookDeletion, NotebookStatus


class MockNotebookAdapter(INotebookAdapter):
    def __init__(self) -> None:
        self._notebooks: dict[str, NotebookStatus] = {}

    def create_notebook(
        self, environment: str, ram_gb: int, gpu_type: str | None = None
    ) -> NotebookStatus:
        del environment, ram_gb, gpu_type  # unused — no real spawner behind this mock
        notebook_id = f"nb-{uuid.uuid4().hex[:8]}"
        status: NotebookStatus = {
            "notebook_id": notebook_id,
            "url": f"http://mock-jupyterhub.local/user/{notebook_id}",
            "active": True,
        }
        self._notebooks[notebook_id] = status
        return status

    def get_notebook_status(self, notebook_id: str) -> NotebookStatus:
        status = self._notebooks.get(notebook_id)
        if status is None:
            raise ValueError(f"Notebook {notebook_id} does not exist")
        return status

    def delete_notebook(self, notebook_id: str) -> NotebookDeletion:
        if notebook_id not in self._notebooks:
            raise ValueError(f"Notebook {notebook_id} does not exist")
        del self._notebooks[notebook_id]
        return {"notebook_id": notebook_id, "deleted": True}
