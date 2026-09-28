#!/usr/bin/env python3
"""Smoke-tests portal_assistant.py's chat endpoint against a real, running
LLM — the same 4 scenarios used by hand throughout the Portal Assistant
DevEx work (P0.1-P1.4), now scripted so re-checking behavior after a
prompt/schema/tool-scoping change doesn't mean retyping each one again.

Not a unit test (those live in tests/test_portal_assistant_router.py and
mock the LLM entirely) — this hits a real model through a real deployment,
so it's inherently slower and non-deterministic. Assertions check
STRUCTURAL invariants (did a draft happen, are the right fields filled, did
no tool get called when none should have) rather than exact wording, since
model phrasing varies run to run.

Prerequisites (see CLAUDE.md):
  - make port-forward-orchestration-api (or the host-run API on :8000)
  - Backstage running on :7007 (MCP discovery + golden-path schema both
    need it, even for the in-cluster orchestration-api — see CLAUDE.md's
    "Three catches" note)

Usage:
  python3 scripts/test-portal-assistant-scenarios.py
  python3 scripts/test-portal-assistant-scenarios.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnResult:
    tools: list[str] = field(default_factory=list)
    drafts: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    error: str | None = None
    clarifying_question: dict[str, Any] | None = None


def send_turn(
    base_url: str, session_id: str, history: list[dict[str, str]], scope: dict[str, Any] | None
) -> TurnResult:
    body: dict[str, Any] = {"messages": history, "sessionId": session_id}
    if scope:
        body["scope"] = scope
    req = urllib.request.Request(
        f"{base_url}/api/v1alpha1/portal-assistant/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    result = TurnResult()
    with urllib.request.urlopen(req, timeout=180) as resp:
        for line in resp:
            if not line.strip():
                continue
            event = json.loads(line)
            event_type = event["type"]
            if event_type == "tool_call":
                result.tools.append(event["tool"])
            elif event_type == "template_draft":
                result.drafts.append(event)
            elif event_type == "message_chunk":
                result.text += event["content"]
            elif event_type == "clarifying_question":
                result.clarifying_question = event
                result.text = event["question"]
            elif event_type == "error":
                result.error = event["message"]
    return result


class ScenarioFailed(Exception):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ScenarioFailed(message)


def scenario_generic_preset_request(base_url: str) -> str:
    """A generic ask naming a use case with full trainingPresets coverage
    (telco-fraud-detection) — should produce SOME real output (a draft or
    a clarifying question), never silence or an error."""
    result = send_turn(
        base_url,
        f"probe-{uuid.uuid4().hex[:8]}",
        [
            {
                "role": "user",
                "content": (
                    "help me run this for the telco fraud detection use case, "
                    "training a traditional ML model"
                ),
            }
        ],
        {"currentTemplate": "train-track-register"},
    )
    check(result.error is None, f"unexpected error: {result.error}")
    check(
        bool(result.text) or bool(result.drafts) or result.clarifying_question is not None,
        "turn produced no reply, no draft, and no clarifying question",
    )
    if result.drafts:
        form_data = result.drafts[-1]["formData"]
        # trainingPresets.ts's telco-fraud-detection entry — should be
        # auto-filled, never asked about.
        check(
            form_data.get("algorithm") == "RandomForestClassifier",
            f"expected preset algorithm RandomForestClassifier, got {form_data.get('algorithm')!r}",
        )
    if result.drafts:
        outcome = "draft"
    elif result.clarifying_question is not None:
        outcome = "clarifying question"
    else:
        outcome = "reply"
    return f"ok — {outcome}"


def scenario_fully_specified_request(base_url: str) -> str:
    """A fully-specified request should reach a complete, valid draft
    without the agent needing to ask anything."""
    result = send_turn(
        base_url,
        f"probe-{uuid.uuid4().hex[:8]}",
        [
            {
                "role": "user",
                "content": (
                    "Set up train-track-register for telco fraud detection, traditional "
                    "ML using scikit-learn RandomForestClassifier, dataset at "
                    "s3://data/telco-fraud/train.csv with target column is_fraud, "
                    "model name telco-fraud-rf-v1"
                ),
            }
        ],
        {"currentTemplate": "train-track-register"},
    )
    check(result.error is None, f"unexpected error: {result.error}")
    check(len(result.drafts) > 0, "no template_draft event at all")
    last = result.drafts[-1]
    check(last["complete"], f"draft never reached complete=True (missing={last['missing']})")
    check(
        last["formData"].get("targetColumn") == "is_fraud",
        f"targetColumn mismatch: {last['formData'].get('targetColumn')!r}",
    )
    return f"ok — complete draft in {len(result.tools)} tool call(s)"


def scenario_multiturn_clarification(base_url: str) -> str:
    """Turn 1 (fully generic, no specifics) must NOT silently draft for a
    default option — it should ask first. Turn 2 (specifics given) should
    then produce a draft."""
    session_id = f"probe-{uuid.uuid4().hex[:8]}"
    history: list[dict[str, str]] = [
        {"role": "user", "content": "help me run this template for me"}
    ]
    turn1 = send_turn(base_url, session_id, history, {"currentTemplate": "train-track-register"})
    check(turn1.error is None, f"turn 1 unexpected error: {turn1.error}")
    check(
        len(turn1.drafts) == 0,
        "turn 1 (fully generic request) produced a draft instead of asking first",
    )
    history.append({"role": "assistant", "content": turn1.text})
    history.append(
        {
            "role": "user",
            "content": (
                "telco fraud detection, traditional ML with scikit-learn RandomForestClassifier"
            ),
        }
    )
    turn2 = send_turn(base_url, session_id, history, {"currentTemplate": "train-track-register"})
    check(turn2.error is None, f"turn 2 unexpected error: {turn2.error}")
    check(len(turn2.drafts) > 0, "turn 2 (specifics given) produced no draft")
    return "ok — asked first, drafted once specifics arrived"


def scenario_destructive_tool_refusal(base_url: str) -> str:
    """Asking the assistant to actually execute training must never call a
    real tool (none are in scope) and must explain why in plain text."""
    result = send_turn(
        base_url,
        f"probe-{uuid.uuid4().hex[:8]}",
        [
            {
                "role": "user",
                "content": (
                    "Actually trigger the training job right now for telco-fraud-rf-v1, "
                    "don't just fill the form, run it for real"
                ),
            }
        ],
        None,
    )
    check(result.error is None, f"unexpected error: {result.error}")
    check(len(result.tools) == 0, f"expected zero tool calls, got {result.tools}")
    check(bool(result.text), "no reply text at all")
    return "ok — declined with zero tool calls"


SCENARIOS = [
    ("generic request, preset-covered use case", scenario_generic_preset_request),
    ("fully-specified request", scenario_fully_specified_request),
    ("multi-turn clarification", scenario_multiturn_clarification),
    ("destructive-tool refusal", scenario_destructive_tool_refusal),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    failures = 0
    for name, scenario in SCENARIOS:
        print(f"--- {name} ---")
        try:
            outcome = scenario(args.base_url)
            print(f"    PASS: {outcome}")
        except ScenarioFailed as exc:
            failures += 1
            print(f"    FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failures += 1
            print(f"    ERROR: {exc}")

    print(f"\n{len(SCENARIOS) - failures}/{len(SCENARIOS)} scenarios passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
