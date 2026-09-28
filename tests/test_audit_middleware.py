"""main.py's `_audit_action` path mapping and `audit_middleware`'s
X-Actor-Ref forwarding.

Regression coverage for two bugs found while wiring Backstage's Scaffolder
actions to forward the real initiator identity:

1. `_audit_action` mislabeled every rollback as "model.promote", since
   `/models/{name}/promote-rollback` contains the substring "/promote" and
   that check ran first.
2. The middleware always recorded the generic "orchestration-api"
   service-account actor, because the Bearer token on these requests is a
   service credential (not a delegated end-user token) — Backstage now
   forwards the real actor via `X-Actor-Ref`, and the middleware reads it.
"""

import pytest
from audit.events import read_audit_events
from main import _audit_action, audit_middleware


def test_audit_action_maps_known_mutating_paths() -> None:
    assert _audit_action("/trigger-training") == "training.trigger"
    assert _audit_action("/models/register") == "model.register"
    assert _audit_action("/deploy-model/prepare") == "model.deploy"
    assert _audit_action("/setup-monitoring") == "monitoring.setup"
    assert _audit_action("/rag/ingest") == "rag.ingest"
    assert _audit_action("/rag/activate") == "rag.activate"
    assert _audit_action("/llm-deploy/prepare") == "llm.serve"
    assert _audit_action("/notebooks") == "notebook.create"
    assert _audit_action("/prompts/churn-summary/activate") == "prompt.activate"


def test_audit_action_distinguishes_promote_from_rollback() -> None:
    assert _audit_action("/models/fraud-detection/promote") == "model.promote"
    assert _audit_action("/models/fraud-detection/promote-rollback") == "model.rollback"
    assert _audit_action("/models/fraud-detection/promote/confirm") == "model.promote"


def test_audit_action_returns_none_for_unmapped_paths() -> None:
    assert _audit_action("/models/fraud-detection/summary") is None
    assert _audit_action("/costs/estimate") is None
    assert _audit_action("/policy-check") is None


class _FakeUrl:
    def __init__(self, path: str) -> None:
        self.path = path


class _FakeRequest:
    def __init__(self, method: str, path: str, headers: dict[str, str] | None = None) -> None:
        self.method = method
        self.url = _FakeUrl(path)
        self.headers = headers or {}


class _FakeResponse:
    status_code = 200


@pytest.mark.asyncio
async def test_audit_middleware_uses_actor_ref_header_when_present() -> None:
    before = len(read_audit_events())
    request = _FakeRequest("POST", "/models/register", {"x-actor-ref": "user:default/jane"})

    async def call_next(_req: object) -> _FakeResponse:
        return _FakeResponse()

    await audit_middleware(request, call_next)

    events = read_audit_events()
    assert len(events) == before + 1
    actor = events[-1]["actor"]
    assert isinstance(actor, dict)
    assert actor["id"] == "user:default/jane"
    assert actor["type"] == "user"


@pytest.mark.asyncio
async def test_audit_middleware_defaults_actor_when_header_absent() -> None:
    before = len(read_audit_events())
    request = _FakeRequest("POST", "/setup-monitoring")

    async def call_next(_req: object) -> _FakeResponse:
        return _FakeResponse()

    await audit_middleware(request, call_next)

    events = read_audit_events()
    assert len(events) == before + 1
    actor = events[-1]["actor"]
    assert isinstance(actor, dict)
    assert actor["id"] == "orchestration-api"
    assert actor["type"] == "service-account"


@pytest.mark.asyncio
async def test_audit_middleware_skips_unmapped_paths() -> None:
    before = len(read_audit_events())
    request = _FakeRequest("POST", "/costs/estimate")

    async def call_next(_req: object) -> _FakeResponse:
        return _FakeResponse()

    await audit_middleware(request, call_next)

    assert len(read_audit_events()) == before
