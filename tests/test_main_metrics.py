"""Regression test for main.py's /metrics endpoint.

`feast` (imported transitively by several routers via
adapters.factory.get_feature_store_adapter) sets `PROMETHEUS_MULTIPROC_DIR`
as an import-time side effect. That env var makes
prometheus_fastapi_instrumentator's /metrics handler switch to reading
multiprocess `.db` files instead of this single-process app's own
in-memory registry — files this app never writes — so /metrics silently
returned an empty body. main.py works around it by popping the env var
(after every router, including the feast-importing ones, has already been
imported) right before calling `Instrumentator().expose(app)`.

This deliberately does NOT set `PROMETHEUS_MULTIPROC_DIR` itself before
importing `main` — doing so would pick `prometheus_client`'s multiprocess
`ValueClass` at the wrong time and no longer reflect the real app's actual
import order (prometheus_client loads, via `prometheus_fastapi_instrumentator`,
before any router/feast import happens inside main.py, since `main.py` —
the real uvicorn entrypoint — is the first thing to ever touch
prometheus_client in a real process). Importing the real `main` module,
unmodified, is what actually reproduces the original bug.

Running under the full test suite, pytest collects (imports) every test
module before running any of them (see tests/conftest.py's own docstring
for other instances of this) — some other test file's collection-time
import chain reaches `feast` before this file's `import main` runs,
already setting PROMETHEUS_MULTIPROC_DIR and locking in
`prometheus_client.values.ValueClass` as its multiprocess mmap-backed
class. That's a test-collection-order artifact, not something that can
happen for the real `main:app` entrypoint (nothing imports it before
uvicorn does) — so it's reset here to match that real-world guarantee,
the same way conftest.py resets other import-order state.
"""

import prometheus_client.values
from fastapi.testclient import TestClient


def test_metrics_endpoint_returns_real_content() -> None:
    prometheus_client.values.ValueClass = prometheus_client.values.MutexValue

    import main

    with TestClient(main.app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    # The default process/gc collectors are always present on the normal
    # in-memory registry — a multiprocess-mode read (the bug) returns
    # neither these nor anything else.
    assert "python_gc_objects_collected_total" in response.text
