"""services/orchestration-api/ttl_cache.py"""

import pytest
from ttl_cache import TTLCache


def test_get_returns_none_for_missing_key() -> None:
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)
    assert cache.get("missing") is None


def test_set_then_get_returns_the_value() -> None:
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)
    cache.set("k", "v")
    assert cache.get("k") == "v"


def test_get_returns_none_once_ttl_elapses(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    monkeypatch.setattr("ttl_cache.time.monotonic", lambda: now[0])
    cache: TTLCache[str] = TTLCache(ttl_seconds=30)
    cache.set("k", "v")
    now[0] += 30
    assert cache.get("k") is None


def test_get_still_hits_just_under_the_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    monkeypatch.setattr("ttl_cache.time.monotonic", lambda: now[0])
    cache: TTLCache[str] = TTLCache(ttl_seconds=30)
    cache.set("k", "v")
    now[0] += 29.999
    assert cache.get("k") == "v"


def test_get_or_set_computes_once_on_a_cache_hit() -> None:
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)
    calls = []

    def compute() -> str:
        calls.append(1)
        return "v"

    assert cache.get_or_set("k", compute) == "v"
    assert cache.get_or_set("k", compute) == "v"
    assert len(calls) == 1


def test_clear_forces_recomputation() -> None:
    cache: TTLCache[str] = TTLCache(ttl_seconds=60)
    cache.set("k", "v")
    cache.clear()
    assert cache.get("k") is None
