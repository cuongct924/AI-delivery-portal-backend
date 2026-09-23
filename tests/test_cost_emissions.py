"""Golden-path steps record real cost events into the ledger, and the observer
cost endpoint surfaces them. tests/conftest.py redirects COST_LEDGER_PATH to a
fresh temp file before any router imports, so the ledger starts clean."""

import uuid
from unittest.mock import patch

from costs.pricing import CPU_HOUR_USD, gpu_hour_price, train_estimate_hours
from routers.observer_costs import get_costs
from routers.prompts import (
    DraftPromptRequest,
    EvaluatePromptRequest,
    PromptEvalCase,
    draft_prompt,
    evaluate_prompt,
)

from adapters.factory import get_cost_adapter

_WINDOW = ("2000-01-01T00:00:00+00:00", "2100-01-01T00:00:00+00:00")


def _ledger_entries(**filters):
    return get_cost_adapter().query_costs(*_WINDOW, **filters)


def test_evaluate_prompt_records_a_real_gate_cost() -> None:
    name = f"cost-eval-{uuid.uuid4().hex[:8]}"
    draft_prompt(DraftPromptRequest(name=name, persona="P", content="sys"))

    with (
        patch("routers.prompts.llm_gateway_adapter") as gateway,
        patch("routers.prompts.judge_response") as judge,
        patch("routers.prompts.evaluate_gate") as gate,
    ):
        gateway.chat_completion.return_value = {
            "choices": [{"message": {"content": "a"}}],
            "usage": {"total_tokens": 100},
            "response_cost_usd": 0.002,
        }
        judge.return_value = {}
        gate.return_value = {"passed": True}
        evaluate_prompt(
            name,
            EvaluatePromptRequest(version="1", eval_cases=[PromptEvalCase(question="q")]),
        )

    entries = _ledger_entries(stage="gate", artifact_kind="prompt")
    match = [e for e in entries if e["artifact_id"] == name]
    assert match, "expected a gate cost event for the evaluated prompt"
    assert match[0]["cost_usd"] == 0.002
    assert match[0]["source"] == "litellm"
    assert match[0]["unit"] == "token"


def test_observer_costs_surfaces_ledger_entries() -> None:
    name = f"cost-obs-{uuid.uuid4().hex[:8]}"
    draft_prompt(DraftPromptRequest(name=name, persona="P", content="sys"))

    with (
        patch("routers.prompts.llm_gateway_adapter") as gateway,
        patch("routers.prompts.judge_response") as judge,
        patch("routers.prompts.evaluate_gate") as gate,
    ):
        gateway.chat_completion.return_value = {
            "choices": [{"message": {"content": "a"}}],
            "usage": {"total_tokens": 10},
            "response_cost_usd": 0.001,
        }
        judge.return_value = {}
        gate.return_value = {"passed": True}
        evaluate_prompt(
            name,
            EvaluatePromptRequest(version="1", eval_cases=[PromptEvalCase(question="q")]),
        )

    response = get_costs(
        namespace="default",
        environment="production",
        startTime="2000-01-01T00:00:00+00:00",
        endTime="2100-01-01T00:00:00+00:00",
    )
    ledger_items = [i for i in response.items if i.artifact == name]
    assert ledger_items, "expected the ledger entry to appear in the observer cost API"
    assert ledger_items[0].stage == "gate"
    # A token entry is routed to tokenCost (not cpuCost) and carries usage, so
    # the frontend's AI cost model and unit economics are populated.
    assert ledger_items[0].tokenCost == 0.001
    assert ledger_items[0].cpuCost == 0.0
    assert ledger_items[0].usage == {"tokens": 10}


def test_pricing_helpers() -> None:
    assert gpu_hour_price("H100") == 4.50
    assert gpu_hour_price("unknown") == 0.80  # falls back to L4
    # More epochs / trials price a longer run.
    assert train_estimate_hours(10, 1) > train_estimate_hours(1, 1)
    assert train_estimate_hours(None, None) > 0
    assert CPU_HOUR_USD > 0
