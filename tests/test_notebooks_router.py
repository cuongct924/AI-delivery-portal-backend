"""services/orchestration-api/routers/notebooks.py — same pattern as the
other router tests: patches the module-level `notebook_adapter` singleton and
calls route functions directly."""

from unittest.mock import patch

import pytest
from fastapi import HTTPException
from routers.notebooks import (
    CreateNotebookRequest,
    create_notebook,
    delete_notebook,
    get_notebook,
    list_notebooks,
    start_notebook,
    stop_notebook,
)

_STATUS = {
    "notebook_id": "nb-1234",
    "url": "http://jupyter.test/user/nb-1234",
    "active": True,
    "environment": "pytorch-cuda",
    "cpu_cores": 4,
    "ram_gb": 16,
    "gpu_type": "t4",
    "gpu_count": 1,
    "storage_gb": 50,
    "idle_timeout_minutes": 30,
    "created_at": "2026-01-01T00:00:00",
    "last_activity_at": "2026-01-01T00:00:00",
}


def test_create_notebook_forwards_the_full_resource_profile() -> None:
    request = CreateNotebookRequest(
        environment="pytorch-cuda",
        cpu_cores=4,
        ram_gb=16,
        gpu_type="t4",
        gpu_count=1,
        storage_gb=50,
        idle_timeout_minutes=30,
    )
    with patch("routers.notebooks.notebook_adapter") as mock_adapter:
        mock_adapter.create_notebook.return_value = _STATUS
        response = create_notebook(request, user={})

    mock_adapter.create_notebook.assert_called_once_with(
        "pytorch-cuda",
        16,
        "t4",
        cpu_cores=4,
        gpu_count=1,
        storage_gb=50,
        idle_timeout_minutes=30,
    )
    assert response.notebook_id == "nb-1234"
    assert response.url == "http://jupyter.test/user/nb-1234"


def test_get_notebook_returns_404_for_unknown_id() -> None:
    with patch("routers.notebooks.notebook_adapter") as mock_adapter:
        mock_adapter.get_notebook_status.side_effect = ValueError("does not exist")
        with pytest.raises(HTTPException) as exc:
            get_notebook("nb-unknown", user={})

    assert exc.value.status_code == 404


def test_list_notebooks_maps_every_status() -> None:
    with patch("routers.notebooks.notebook_adapter") as mock_adapter:
        mock_adapter.list_notebooks.return_value = [_STATUS]
        response = list_notebooks(user={})

    assert [n.notebook_id for n in response] == ["nb-1234"]


def test_stop_then_start_round_trips_through_the_adapter() -> None:
    stopped = {**_STATUS, "active": False, "url": None}
    with patch("routers.notebooks.notebook_adapter") as mock_adapter:
        mock_adapter.stop_notebook.return_value = stopped
        assert stop_notebook("nb-1234", user={}).active is False

        mock_adapter.start_notebook.return_value = _STATUS
        assert start_notebook("nb-1234", user={}).active is True


def test_delete_notebook_returns_deletion_result() -> None:
    with patch("routers.notebooks.notebook_adapter") as mock_adapter:
        mock_adapter.delete_notebook.return_value = {
            "notebook_id": "nb-1234",
            "deleted": True,
        }
        response = delete_notebook("nb-1234", user={})

    assert response.deleted is True
