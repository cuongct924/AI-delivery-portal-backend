"""services/orchestration-api/routers/golden_paths.py — patches
catalog_client.list_golden_path_templates/get_golden_path_template as
imported into the router module, and calls route functions directly (same
pattern as tests/test_models_router.py), no FastAPI TestClient needed.
"""

from unittest.mock import patch

import pytest
from fastapi import HTTPException
from routers.golden_paths import get_golden_path, list_golden_paths


def test_list_golden_paths_returns_catalog_templates() -> None:
    with patch(
        "routers.golden_paths.list_golden_path_templates",
        return_value=[
            {
                "name": "train-track-register",
                "title": "Train & Register Model",
                "description": "Trains a model.",
                "tags": ["mlops"],
            }
        ],
    ):
        result = list_golden_paths(user={})

    assert len(result) == 1
    assert result[0].name == "train-track-register"
    assert result[0].tags == ["mlops"]


def test_get_golden_path_returns_full_detail() -> None:
    with patch(
        "routers.golden_paths.get_golden_path_template",
        return_value={
            "name": "train-track-register",
            "title": "Train & Register Model",
            "description": "Trains a model.",
            "tags": ["mlops"],
            "parameters": [
                {"name": "modelName", "title": "Model name", "description": "", "required": True}
            ],
            "steps": [{"name": "Train", "action": "orchestration:trigger-training"}],
        },
    ):
        result = get_golden_path("train-track-register", user={})

    assert result.title == "Train & Register Model"
    assert result.parameters[0].name == "modelName"
    assert result.steps[0].action == "orchestration:trigger-training"


def test_get_golden_path_raises_404_when_not_found() -> None:
    with (
        patch("routers.golden_paths.get_golden_path_template", return_value=None),
        pytest.raises(HTTPException) as exc_info,
    ):
        get_golden_path("does-not-exist", user={})

    assert exc_info.value.status_code == 404
