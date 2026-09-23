"""Tests services/orchestration-api/evaluations/llm_judge.py — mocks
LiteLLMGatewayAdapter (same patch-the-adapter-class pattern as
tests/test_evaluate_drift.py) so no real LiteLLM proxy is required.
"""

from unittest.mock import patch

from evaluations.llm_judge import judge_response


def test_judge_response_parses_openai_compatible_content() -> None:
    litellm_response = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"safety": 9, "correctness": 8, "relevance": 7, "reasoning": "ok"}'
                    )
                }
            }
        ]
    }

    with patch("evaluations.llm_judge.LiteLLMGatewayAdapter") as mock_adapter_cls:
        mock_adapter_cls.return_value.chat_completion.return_value = litellm_response
        result = judge_response("What is 2+2?", "4")

    assert result == {"safety": 9, "correctness": 8, "relevance": 7, "reasoning": "ok"}


def test_judge_response_sends_system_and_user_messages() -> None:
    litellm_response = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"safety": 10, "correctness": 10, "relevance": 10, "reasoning": "x"}'
                    )
                }
            }
        ]
    }

    with patch("evaluations.llm_judge.LiteLLMGatewayAdapter") as mock_adapter_cls:
        mock_adapter_cls.return_value.chat_completion.return_value = litellm_response
        judge_response("question", "answer")

    _, kwargs = mock_adapter_cls.return_value.chat_completion.call_args
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["messages"][0]["role"] == "system"
    assert "question" in kwargs["messages"][1]["content"]


def test_judge_response_with_context_asks_for_faithfulness() -> None:
    litellm_response = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"safety": 9, "correctness": 8, "relevance": 7, '
                        '"faithfulness": 6, "reasoning": "ok"}'
                    )
                }
            }
        ]
    }

    with patch("evaluations.llm_judge.LiteLLMGatewayAdapter") as mock_adapter_cls:
        mock_adapter_cls.return_value.chat_completion.return_value = litellm_response
        result = judge_response("question", "answer", context="retrieved chunk")

    assert result.get("faithfulness") == 6
    _, kwargs = mock_adapter_cls.return_value.chat_completion.call_args
    assert "faithfulness" in kwargs["messages"][0]["content"]
    assert "retrieved chunk" in kwargs["messages"][1]["content"]


def test_judge_response_without_context_never_mentions_faithfulness() -> None:
    litellm_response = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"safety": 9, "correctness": 8, "relevance": 7, "reasoning": "ok"}'
                    )
                }
            }
        ]
    }

    with patch("evaluations.llm_judge.LiteLLMGatewayAdapter") as mock_adapter_cls:
        mock_adapter_cls.return_value.chat_completion.return_value = litellm_response
        result = judge_response("question", "answer")

    assert "faithfulness" not in result
    _, kwargs = mock_adapter_cls.return_value.chat_completion.call_args
    assert "faithfulness" not in kwargs["messages"][0]["content"]
