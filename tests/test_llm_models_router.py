"""services/orchestration-api/routers/llm_models.py — patches the
module-level `llm_gateway_adapter` singleton and calls the route function
directly, same pattern as tests/test_prompts_router.py's evaluate_prompt
test.
"""

from unittest.mock import patch

from routers.llm_models import list_llm_model_names


def test_list_llm_model_names_extracts_id_from_each_model() -> None:
    with patch("routers.llm_models.llm_gateway_adapter") as mock_gateway:
        mock_gateway.list_models.return_value = [
            {"id": "claude-sonnet-5", "object": "model"},
            {"id": "llama3.1-local", "object": "model"},
        ]
        response = list_llm_model_names()

    assert response.names == ["claude-sonnet-5", "llama3.1-local"]


def test_list_llm_model_names_returns_empty_when_gateway_has_no_models() -> None:
    with patch("routers.llm_models.llm_gateway_adapter") as mock_gateway:
        mock_gateway.list_models.return_value = []
        response = list_llm_model_names()

    assert response.names == []
