"""In-memory Idempotency-Key store for POST endpoints that create a
resource with a non-deterministic name (e.g. a WorkflowRun named
`{template}-{timestamp_ms}`) — a retried request (Scaffolder step retry,
double-click, network timeout) must return the original result instead of
triggering a second real side effect.

Per-process, TTL-bounded — same trust boundary as the other module-level
dicts in routers/models.py (`_TRAINING_MODEL_NAMES` etc.): this service has
no shared cache between replicas, and a restart losing in-flight keys is an
acceptable tradeoff (worst case, a duplicate on a 1-in-a-million restart
race) for a Golden Path Scaffolder runs at most once per click.
"""

import time
from collections.abc import Callable
from typing import cast

_TTL_SECONDS = 24 * 60 * 60

_store: dict[str, tuple[float, object]] = {}


def get_or_compute[T](key: str | None, compute: Callable[[], T]) -> T:
    """Runs `compute()` and caches its result under `key` for `_TTL_SECONDS`.
    A repeat call with the same key returns the cached result without
    calling `compute()` again. A non-`str` key (no Idempotency-Key header
    sent — FastAPI resolves that to `None`; a route handler called directly
    with no argument, bypassing FastAPI's dependency injection, gets its
    `Header(default=None)` sentinel object instead) always computes fresh —
    same as today's behavior."""
    if not isinstance(key, str):
        return compute()
    _evict_expired()
    cached = _store.get(key)
    if cached is not None:
        return cast(T, cached[1])
    result = compute()
    _store[key] = (time.monotonic() + _TTL_SECONDS, result)
    return result


def _evict_expired() -> None:
    now = time.monotonic()
    expired = [key for key, (expiry, _) in _store.items() if expiry < now]
    for key in expired:
        del _store[key]
