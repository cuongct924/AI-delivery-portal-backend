"""Tests for adapters/mock_eval_result_adapter.py — in-memory adapter for
LLM-as-a-judge evaluation results."""

from datetime import datetime

from adapters.mock_eval_result_adapter import MockEvalResultAdapter


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
