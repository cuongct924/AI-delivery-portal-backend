"""services/orchestration-api/routers/eval_sets.py — backed by a real
JsonFileVersionRegistryAdapter (same registry file as rag-index, different
`kind`) — tests/conftest.py redirects LLMOPS_REGISTRY_PATH to a fresh temp
file before any router module imports, same as tests/test_rag_router.py.
"""

import pytest
from fastapi import HTTPException
from routers.eval_sets import (
    DraftEvalSetRequest,
    draft_eval_set,
    get_eval_set_version,
    get_latest_eval_set_version,
    list_eval_set_names,
    list_eval_set_versions,
)


# Declared first: the registry file is shared for the whole test session
# (conftest.py redirects it once), so this is the only test that still sees
# an empty eval-set kind before any draft below registers a name.
def test_get_latest_eval_set_version_404_hints_when_registry_empty() -> None:
    with pytest.raises(HTTPException) as exc_info:
        get_latest_eval_set_version("eval set name")

    assert "No eval sets registered yet" in exc_info.value.detail


def test_draft_eval_set_registers_first_version() -> None:
    response = draft_eval_set(DraftEvalSetRequest(name="idp-basics", questions=["q1", "q2"]))

    assert response.name == "idp-basics"
    assert response.version == "1"
    assert response.questions == ["q1", "q2"]
    assert "idp-basics" in list_eval_set_names().names


def test_draft_eval_set_increments_version_for_same_name() -> None:
    draft_eval_set(DraftEvalSetRequest(name="incrementing-set", questions=["q1"]))
    second = draft_eval_set(DraftEvalSetRequest(name="incrementing-set", questions=["q1", "q2"]))

    assert second.version == "2"
    assert list_eval_set_versions("incrementing-set").versions == ["1", "2"]


def test_get_eval_set_version_returns_the_registered_questions() -> None:
    draft_eval_set(DraftEvalSetRequest(name="fetch-target", questions=["a", "b", "c"]))

    result = get_eval_set_version("fetch-target", "1")
    assert result.questions == ["a", "b", "c"]


def test_get_eval_set_version_raises_404_for_missing_version() -> None:
    with pytest.raises(HTTPException) as exc_info:
        get_eval_set_version("does-not-exist", "1")
    assert exc_info.value.status_code == 404


def test_get_latest_eval_set_version_returns_the_highest_version() -> None:
    draft_eval_set(DraftEvalSetRequest(name="latest-target", questions=["v1"]))
    draft_eval_set(DraftEvalSetRequest(name="latest-target", questions=["v2-a", "v2-b"]))

    result = get_latest_eval_set_version("latest-target")
    assert result.version == "2"
    assert result.questions == ["v2-a", "v2-b"]


def test_get_latest_eval_set_version_raises_404_when_never_drafted() -> None:
    with pytest.raises(HTTPException) as exc_info:
        get_latest_eval_set_version("never-drafted")
    assert exc_info.value.status_code == 404


def test_get_latest_eval_set_version_404_lists_available_names() -> None:
    draft_eval_set(DraftEvalSetRequest(name="idp-basics", questions=["q1"]))

    with pytest.raises(HTTPException) as exc_info:
        get_latest_eval_set_version("eval set name")

    assert exc_info.value.status_code == 404
    assert "idp-basics" in exc_info.value.detail
