"""AI Notebook API — provisions and manages per-user JupyterHub servers via
`INotebookAdapter` (JupyterHubAdapter, or MockNotebookAdapter under
USE_MOCK_NOTEBOOK). This is the interactive counterpart to Golden Path #1's
declarative Argo training: the Dev writes/runs code in the notebook, logs the
run to MLflow, then registers the model through the normal register path.

The Portal only provisions/manages here — the editing surface stays
JupyterHub's own web IDE, never re-implemented in Backstage. The adapter
existed but had no HTTP surface; this router is that surface, backing the
Portal's "Open AI Notebook" Scaffolder template.
"""

from auth.thunder import get_current_user
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from adapters.factory import get_notebook_adapter

router = APIRouter(prefix="/notebooks", tags=["notebooks"])

notebook_adapter = get_notebook_adapter()


class CreateNotebookRequest(BaseModel):
    # JupyterHub KubeSpawner profile option — e.g. "pytorch-cuda",
    # "tensorflow-cuda", "sklearn-cpu". Must match a profile_list entry on
    # the real deployment.
    environment: str
    cpu_cores: int = 2
    ram_gb: int = 8
    # "none" (CPU-only) or a GPU type the spawner profile offers, e.g. "t4".
    gpu_type: str | None = None
    gpu_count: int = 1
    storage_gb: int = 20
    # Auto-shutdown after this many idle minutes (0 = never).
    idle_timeout_minutes: int = 60


class NotebookResponse(BaseModel):
    notebook_id: str
    url: str | None
    active: bool
    environment: str
    cpu_cores: int
    ram_gb: int
    gpu_type: str | None
    gpu_count: int
    storage_gb: int
    idle_timeout_minutes: int
    created_at: str
    last_activity_at: str | None


class DeleteNotebookResponse(BaseModel):
    notebook_id: str
    deleted: bool


def _get_or_404(notebook_id: str) -> NotebookResponse:
    try:
        return NotebookResponse(**notebook_adapter.get_notebook_status(notebook_id))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("", response_model=NotebookResponse)
def create_notebook(
    request: CreateNotebookRequest, user: dict = Depends(get_current_user)
) -> NotebookResponse:
    del user
    status = notebook_adapter.create_notebook(
        request.environment,
        request.ram_gb,
        request.gpu_type,
        cpu_cores=request.cpu_cores,
        gpu_count=request.gpu_count,
        storage_gb=request.storage_gb,
        idle_timeout_minutes=request.idle_timeout_minutes,
    )
    return NotebookResponse(**status)


@router.get("", response_model=list[NotebookResponse])
def list_notebooks(user: dict = Depends(get_current_user)) -> list[NotebookResponse]:
    del user
    return [NotebookResponse(**status) for status in notebook_adapter.list_notebooks()]


@router.get("/{notebook_id}", response_model=NotebookResponse)
def get_notebook(notebook_id: str, user: dict = Depends(get_current_user)) -> NotebookResponse:
    del user
    return _get_or_404(notebook_id)


@router.post("/{notebook_id}/start", response_model=NotebookResponse)
def start_notebook(notebook_id: str, user: dict = Depends(get_current_user)) -> NotebookResponse:
    del user
    try:
        status = notebook_adapter.start_notebook(notebook_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return NotebookResponse(**status)


@router.post("/{notebook_id}/stop", response_model=NotebookResponse)
def stop_notebook(notebook_id: str, user: dict = Depends(get_current_user)) -> NotebookResponse:
    del user
    try:
        status = notebook_adapter.stop_notebook(notebook_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return NotebookResponse(**status)


@router.delete("/{notebook_id}", response_model=DeleteNotebookResponse)
def delete_notebook(
    notebook_id: str, user: dict = Depends(get_current_user)
) -> DeleteNotebookResponse:
    del user
    try:
        result = notebook_adapter.delete_notebook(notebook_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return DeleteNotebookResponse(**result)
