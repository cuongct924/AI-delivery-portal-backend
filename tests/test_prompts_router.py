"""services/orchestration-api/routers/prompts.py — calls the route functions
directly, no need for a FastAPI TestClient since we're not testing the HTTP/
routing layer. Backed by a real JsonFileVersionRegistryAdapter — tests/
conftest.py redirects LLMOPS_REGISTRY_PATH to a fresh temp file before any
router module imports, so _seed_default_prompts() (run at import time)
always starts from a clean slate for this test session."""

import uuid
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from routers.prompts import (
    ActivatePromptRequest,
    DraftPromptRequest,
    EvaluatePromptRequest,
    PromptEvalCase,
    activate_prompt,
    draft_prompt,
    evaluate_prompt,
    get_prompt,
    get_prompt_active_version,
    list_prompt_names,
    list_prompt_versions,
)


def test_list_prompt_names_returns_seeded_prompts():
    # Superset, not equality — a persona name a prior local test run
    # registered against the real MLflow Prompt Registry (never wiped
    # between runs, unlike the JSON-file rag-index registry) has to stay
    # listed too.
    assert {"mlops", "k8s"} <= set(list_prompt_names().names)


def test_list_prompt_versions_returns_registered_versions():
    # Unique per test run — MLflow prompt versions only ever increment, so
    # a fixed name would accumulate extra versions across repeated local
    # runs and break the exact ["1", "2"] assertion below.
    name = f"versions-target-{uuid.uuid4().hex[:8]}"
    draft_prompt(DraftPromptRequest(name=name, persona="V", content="v1"))
    draft_prompt(DraftPromptRequest(name=name, persona="V", content="v2"))

    assert list_prompt_versions(name).versions == ["1", "2"]


def test_list_prompt_versions_returns_empty_for_unregistered_name():
    assert list_prompt_versions("does-not-exist").versions == []


def test_get_prompt_active_version_returns_active_version():
    draft_prompt(DraftPromptRequest(name="active-target", persona="A", content="sys"))
    activate_prompt("active-target", ActivatePromptRequest(version="1"))

    response = get_prompt_active_version("active-target")
    assert response.name == "active-target"
    assert response.active_version == "1"


def test_get_prompt_active_version_returns_none_when_never_activated():
    draft_prompt(DraftPromptRequest(name="unactivated-target", persona="U", content="sys"))

    assert get_prompt_active_version("unactivated-target").active_version is None


def test_get_prompt_found():
    prompt = get_prompt("mlops-v1")
    assert prompt.persona == "MLOps Assistant"


def test_get_prompt_not_found_raises_404():
    with pytest.raises(HTTPException) as exc_info:
        get_prompt("does-not-exist")
    assert exc_info.value.status_code == 404


def test_draft_prompt_registers_a_new_unactivated_version():
    # Unique per test run — same reasoning as
    # test_list_prompt_versions_returns_registered_versions: MLflow prompt
    # versions only ever increment, so a fixed name's version number isn't
    # stable across repeated local runs against the same MLflow instance.
    name = f"rag-writer-{uuid.uuid4().hex[:8]}"
    request = DraftPromptRequest(name=name, persona="RAG Writer", content="Draft content")
    response = draft_prompt(request)

    assert response.id == f"{name}-v1"
    assert response.version == "1"
    # Drafting registers the name, but doesn't activate the version.
    assert name in list_prompt_names().names
    assert get_prompt_active_version(name).active_version is None


def test_evaluate_prompt_computes_pass_rate_and_forwards_model():
    draft_prompt(DraftPromptRequest(name="eval-target", persona="Eval Target", content="sys"))
    request = EvaluatePromptRequest(
        version="1",
        eval_cases=[PromptEvalCase(question="q1"), PromptEvalCase(question="q2")],
        model="llama-3-8b-self-hosted",
    )
    with (
        patch("routers.prompts.llm_gateway_adapter") as mock_gateway,
        patch("routers.prompts.judge_response") as mock_judge,
        patch("routers.prompts.evaluate_gate") as mock_gate,
    ):
        mock_gateway.chat_completion.return_value = {
            "choices": [{"message": {"content": "an answer"}}],
            "usage": {"total_tokens": 100},
            "response_cost_usd": 0.002,
        }
        mock_judge.return_value = {"safety": 9, "correctness": 9, "relevance": 9}
        mock_gate.side_effect = [{"passed": True}, {"passed": True}]
        response = evaluate_prompt("eval-target", request)

    assert response.passed is True
    assert response.pass_rate == 1.0
    assert response.total_tokens == 200  # 100 per eval_case, 2 eval_cases
    assert response.total_cost_usd == pytest.approx(0.004)
    mock_gateway.chat_completion.assert_any_call(
        model="llama-3-8b-self-hosted",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
        ],
    )


def test_activate_prompt_raises_for_unregistered_version():
    with pytest.raises(ValueError, match="no registered versions"):
        activate_prompt("never-drafted", ActivatePromptRequest(version="1"))
