"""Tests adapters/mock_misc_adapters.py — MockEvalResultAdapter,
MockNotebookAdapter, MockHuggingFaceHubAdapter."""

from datetime import datetime

import pytest

from adapters.ai_platform.mock_misc_adapters import (
    MockEvalResultAdapter,
    MockHuggingFaceHubAdapter,
    MockNotebookAdapter,
)


def test_log_judge_result_and_get_last_failure() -> None:
    adapter = MockEvalResultAdapter()

    # Log a passing result
    adapter.log_judge_result(
        kind="prompt",
        name="my-prompt",
        version="1",
        judge_result={"safety": 9, "correctness": 8, "relevance": 9, "reasoning": "good"},
        passed=True,
    )

    # No failures yet
    assert adapter.get_last_failure_at("prompt", "my-prompt") is None

    # Log a failing result
    adapter.log_judge_result(
        kind="prompt",
        name="my-prompt",
        version="2",
        judge_result={"safety": 5, "correctness": 4, "relevance": 3, "reasoning": "bad"},
        passed=False,
    )

    # Should return the failure timestamp
    failure_time = adapter.get_last_failure_at("prompt", "my-prompt")
    assert failure_time is not None
    assert isinstance(failure_time, datetime)

    # Log another passing result
    adapter.log_judge_result(
        kind="prompt",
        name="my-prompt",
        version="3",
        judge_result={"safety": 9, "correctness": 9, "relevance": 9, "reasoning": "good again"},
        passed=True,
    )

    # Last failure should still be the one from version 2
    failure_time2 = adapter.get_last_failure_at("prompt", "my-prompt")
    assert failure_time2 == failure_time

    # Log another failing result (newer)
    adapter.log_judge_result(
        kind="prompt",
        name="my-prompt",
        version="4",
        judge_result={"safety": 4, "correctness": 3, "relevance": 2, "reasoning": "worse"},
        passed=False,
    )

    # Now last failure should be the newer one
    failure_time3 = adapter.get_last_failure_at("prompt", "my-prompt")
    assert failure_time3 is not None
    assert failure_time is not None
    assert failure_time3 > failure_time


def test_get_last_failure_at_different_kind_and_name() -> None:
    adapter = MockEvalResultAdapter()

    adapter.log_judge_result(
        kind="prompt",
        name="prompt-a",
        version="1",
        judge_result={"safety": 5, "correctness": 4, "relevance": 3},
        passed=False,
    )

    adapter.log_judge_result(
        kind="rag-index",
        name="collection-x",
        version="1",
        judge_result={"safety": 5, "correctness": 4, "relevance": 3},
        passed=False,
    )

    # Each kind/name pair is independent
    assert adapter.get_last_failure_at("prompt", "prompt-a") is not None
    assert adapter.get_last_failure_at("rag-index", "collection-x") is not None
    assert adapter.get_last_failure_at("prompt", "prompt-b") is None
    assert adapter.get_last_failure_at("rag-index", "collection-y") is None


@pytest.fixture
def notebook_adapter() -> MockNotebookAdapter:
    return MockNotebookAdapter()


def test_create_notebook_returns_an_active_notebook(notebook_adapter: MockNotebookAdapter) -> None:
    status = notebook_adapter.create_notebook("pytorch", ram_gb=8, gpu_type="t4")

    assert status["active"] is True
    assert status["url"] is not None


def test_get_notebook_status_returns_what_was_created(
    notebook_adapter: MockNotebookAdapter,
) -> None:
    created = notebook_adapter.create_notebook("pytorch", ram_gb=8)

    fetched = notebook_adapter.get_notebook_status(created["notebook_id"])

    assert fetched == created


def test_get_notebook_status_raises_for_unknown_id(notebook_adapter: MockNotebookAdapter) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        notebook_adapter.get_notebook_status("nb-unknown")


def test_delete_notebook_removes_it(notebook_adapter: MockNotebookAdapter) -> None:
    created = notebook_adapter.create_notebook("pytorch", ram_gb=8)

    result = notebook_adapter.delete_notebook(created["notebook_id"])

    assert result == {"notebook_id": created["notebook_id"], "deleted": True}
    with pytest.raises(ValueError, match="does not exist"):
        notebook_adapter.get_notebook_status(created["notebook_id"])


def test_delete_notebook_raises_for_unknown_id(notebook_adapter: MockNotebookAdapter) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        notebook_adapter.delete_notebook("nb-unknown")


def test_known_gated_model_returns_real_architecture() -> None:
    info = MockHuggingFaceHubAdapter().get_model_info("meta-llama/Llama-3.1-8B-Instruct")
    assert info["exists"] is True
    assert info["is_gated"] is True
    assert info["num_layers"] == 32


def test_known_not_found_model_returns_exists_false() -> None:
    info = MockHuggingFaceHubAdapter().get_model_info("does-not-exist/not-a-real-model")
    assert info["exists"] is False


def test_unknown_model_id_returns_synthetic_ungated_info() -> None:
    info = MockHuggingFaceHubAdapter().get_model_info("some-org/some-model")
    assert info["exists"] is True
    assert info["is_gated"] is False
    assert info["param_count_billion"] == 7.0
