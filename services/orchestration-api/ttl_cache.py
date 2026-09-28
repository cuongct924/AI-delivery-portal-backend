"""Minimal in-process TTL cache for cheap-to-recompute values that would
otherwise mean a network round trip on every single chat message — a
Prompt Registry read, a Catalog schema lookup. Deliberately not a generic
memoize decorator: process-local, unbounded per key, correct only for the
small, fixed set of keys (persona names, template names)
routers/portal_assistant.py actually uses. A short TTL (seconds, not
minutes) is the point — it batches the repeated reads within one
conversation's burst of messages without meaningfully delaying an
in-progress `/prompts/{name}/activate` iteration loop.
"""

import time
from collections.abc import Callable


class TTLCache[T]:
    def __init__(self, ttl_seconds: float) -> None:
        self._ttl_seconds = ttl_seconds
        self._entries: dict[str, tuple[T, float]] = {}

    def get(self, key: str) -> T | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        value, cached_at = entry
        if time.monotonic() - cached_at >= self._ttl_seconds:
            return None
        return value

    def set(self, key: str, value: T) -> None:
        self._entries[key] = (value, time.monotonic())

    def get_or_set(self, key: str, compute: Callable[[], T]) -> T:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute()
        self.set(key, value)
        return value

    def clear(self) -> None:
        self._entries.clear()
