"""Tests services/orchestration-api/idempotency.py."""

import idempotency


def test_none_key_always_computes_fresh() -> None:
    calls = 0

    def compute() -> int:
        nonlocal calls
        calls += 1
        return calls

    assert idempotency.get_or_compute(None, compute) == 1
    assert idempotency.get_or_compute(None, compute) == 2


def test_repeated_key_returns_the_cached_result_without_recomputing() -> None:
    calls = 0

    def compute() -> str:
        nonlocal calls
        calls += 1
        return f"result-{calls}"

    first = idempotency.get_or_compute("task-1", compute)
    second = idempotency.get_or_compute("task-1", compute)

    assert first == "result-1"
    assert second == "result-1"
    assert calls == 1


def test_different_keys_compute_independently() -> None:
    calls = 0

    def compute() -> int:
        nonlocal calls
        calls += 1
        return calls

    assert idempotency.get_or_compute("task-1", compute) == 1
    assert idempotency.get_or_compute("task-2", compute) == 2


def test_expired_entries_are_evicted_and_recomputed() -> None:
    calls = 0

    def compute() -> int:
        nonlocal calls
        calls += 1
        return calls

    idempotency.get_or_compute("task-1", compute)
    expiry, cached = idempotency._store["task-1"]
    idempotency._store["task-1"] = (expiry - 2 * idempotency._TTL_SECONDS, cached)

    assert idempotency.get_or_compute("task-1", compute) == 2
    assert calls == 2
