"""pytest collects (imports) every test module in this directory before
running any of them. Importing torch here first, before pytest reaches a
module that imports xgboost/lightgbm (algorithm_registry.py, transitively:
tests/test_algorithm_registry.py, tests/test_models_router.py, ...), avoids
a macOS-only segfault — loading xgboost/lightgbm's OpenMP runtime before
torch's own crashes on the first CrossEntropyLoss call in this process.
Harmless on Linux (the actual training-image target).

Importing mlflow.pyfunc here first, for the same collection-order reason:
tests/test_mlflow_adapter.py and tests/test_models_router.py both do
`sys.modules.setdefault("mlflow", MagicMock())` at module level (to skip
mlflow's real, heavy import for their own unit tests) — since pytest
collects every module before running any test, that stub otherwise wins the
race and poisons `sys.modules["mlflow"]` for the whole session. Plain
`mlflow.log_metric(...)`-style calls elsewhere degrade harmlessly to a mock
call either way, but `pyfunc_wrapper.GenericPyfuncWrapper` subclasses
`mlflow.pyfunc.PythonModel` — a mocked, non-type attribute can't be
subclassed correctly, so it needs the real module loaded first.

Setting QDRANT_REGISTRY_COLLECTION before anything imports routers.prompts/
routers.rag/routers.chat/routers.eval_sets, for a similar reason: each of
those constructs a module-level QdrantVersionRegistryAdapter() singleton at
import time, which reads this env var in __init__. Left unset, all of them
would share the real `_llmops_registry` collection — which, unlike the old
JSON file, persists across runs and would pollute the next one
(tests/test_eval_sets_router.py's empty-registry assertion would silently
start reading stale state). A unique collection per test session gives every
run a clean slate; not cleaned up afterwards since it's outside the repo."""

import importlib.util
import os
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType

import pytest

os.environ.setdefault("QDRANT_REGISTRY_COLLECTION", f"_llmops_registry_test_{uuid.uuid4().hex[:8]}")

# Same reasoning as QDRANT_REGISTRY_COLLECTION above: routers.costs constructs a
# module-level JsonFileCostLedgerAdapter() at import time, which reads this env
# var in __init__. Redirect it to a fresh temp file so a test run never reads
# or pollutes the real .state/cost-ledger.json.
os.environ.setdefault("COST_LEDGER_PATH", os.path.join(tempfile.mkdtemp(), "cost-ledger.json"))

# Same reasoning again: audit.events reads AUDIT_LOG_PATH at import time.
os.environ.setdefault("AUDIT_LOG_PATH", os.path.join(tempfile.mkdtemp(), "audit-log.json"))

import idempotency  # noqa: E402
import mlflow.pyfunc  # noqa: F401, E402
import torch  # noqa: F401, E402


@pytest.fixture(autouse=True)
def _clear_idempotency_store() -> None:
    """idempotency._store is a process-wide module-level dict — without
    this, a key reused across two test functions (unlikely, but the whole
    point of an Idempotency-Key is to collide on purpose) would leak a
    cached response from one test into another."""
    idempotency._store.clear()


def make_stream(
    *rounds: Sequence[Mapping[str, object]],
) -> Callable[..., AsyncIterator[Mapping[str, object]]]:
    """Builds a `side_effect`-ready factory for a mocked
    `ILLMGatewayAdapter.chat_completion_stream`, replacing the
    non-streaming `mock_gateway.chat_completion.side_effect = [resp1, resp2]`
    pattern (a plain list doesn't work here: `chat_completion_stream` must
    return something `async for`-able per call, not a dict, and Mock's
    list-of-return-values semantics returns each element as-is rather than
    calling it).

    One or more rounds, each a list of delta-chunk dicts; the Nth call to
    the mocked method yields round N's chunks as a fresh async generator
    (clamped to the last round if called more times than rounds given —
    tests only need this when a round DOESN'T end the loop, i.e. it made a
    tool call and a further round follows).

    Usage: `mock_gateway.chat_completion_stream.side_effect =
    make_stream(round1_chunks, round2_chunks)`.
    """
    call_count = 0

    async def _gen(*args: object, **kwargs: object) -> AsyncIterator[Mapping[str, object]]:
        nonlocal call_count
        chunks = rounds[min(call_count, len(rounds) - 1)]
        call_count += 1
        for chunk in chunks:
            yield chunk

    return _gen


def load_module_from_path(alias: str, file_path: Path) -> ModuleType:
    """Load a .py file as a module under a unique `alias` name in
    sys.modules, bypassing normal package/path-based resolution.

    Used for agents/mcp-servers/*/ directories: each contains a
    same-named server.py/token_verifier.py/thunder_client.py, so putting
    two such directories on one global pythonpath/extraPaths would make a
    bare `import server` ambiguous across test files — whichever
    directory happened to be imported first in the session silently wins
    for every test after it, including files that meant to test a
    different server entirely.
    """
    spec = importlib.util.spec_from_file_location(alias, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def load_mcp_server(server_dir: Path, prefix: str) -> ModuleType:
    """Load one agents/mcp-servers/<x>/server.py, with its sibling
    thunder_client.py/token_verifier.py loaded first (under `prefix`-
    namespaced aliases) and briefly bound to their bare names in
    sys.modules so server.py's own `from thunder_client import ...` /
    `from token_verifier import ...` resolve to *this* directory's
    copies — then restored to whatever was cached there before (never
    just deleted), so this doesn't leave a different, already-imported
    server module holding a stale/orphaned reference to a module object
    that's no longer the one a later `import token_verifier` would
    re-resolve to.
    """
    thunder_client = load_module_from_path(
        f"{prefix}_thunder_client", server_dir / "thunder_client.py"
    )
    token_verifier = load_module_from_path(
        f"{prefix}_token_verifier", server_dir / "token_verifier.py"
    )
    prev_thunder_client = sys.modules.get("thunder_client")
    prev_token_verifier = sys.modules.get("token_verifier")
    sys.modules["thunder_client"] = thunder_client
    sys.modules["token_verifier"] = token_verifier
    server = load_module_from_path(f"{prefix}_server", server_dir / "server.py")
    if prev_thunder_client is not None:
        sys.modules["thunder_client"] = prev_thunder_client
    else:
        del sys.modules["thunder_client"]
    if prev_token_verifier is not None:
        sys.modules["token_verifier"] = prev_token_verifier
    else:
        del sys.modules["token_verifier"]
    return server
