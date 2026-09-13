"""Mock adapter for INotebookAdapter — in-memory stand-in for JupyterHub.
Not wired into any router yet, same as JupyterHubAdapter itself.
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
