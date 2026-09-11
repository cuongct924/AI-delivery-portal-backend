"""Tests adapters/mock_notebook_adapter.py."""

import pytest

from adapters.mock_notebook_adapter import MockNotebookAdapter


@pytest.fixture
def adapter() -> MockNotebookAdapter:
    return MockNotebookAdapter()


def test_create_notebook_returns_an_active_notebook(adapter: MockNotebookAdapter) -> None:
    status = adapter.create_notebook("pytorch", ram_gb=8, gpu_type="t4")

    assert status["active"] is True
    assert status["url"] is not None


def test_get_notebook_status_returns_what_was_created(adapter: MockNotebookAdapter) -> None:
    created = adapter.create_notebook("pytorch", ram_gb=8)

    fetched = adapter.get_notebook_status(created["notebook_id"])

    assert fetched == created


def test_get_notebook_status_raises_for_unknown_id(adapter: MockNotebookAdapter) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        adapter.get_notebook_status("nb-unknown")


def test_delete_notebook_removes_it(adapter: MockNotebookAdapter) -> None:
    created = adapter.create_notebook("pytorch", ram_gb=8)

    result = adapter.delete_notebook(created["notebook_id"])

    assert result == {"notebook_id": created["notebook_id"], "deleted": True}
    with pytest.raises(ValueError, match="does not exist"):
        adapter.get_notebook_status(created["notebook_id"])


def test_delete_notebook_raises_for_unknown_id(adapter: MockNotebookAdapter) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        adapter.delete_notebook("nb-unknown")
