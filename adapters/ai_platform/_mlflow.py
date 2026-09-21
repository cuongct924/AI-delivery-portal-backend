"""Shared MLflow tracking-URI resolution for mlflow_adapter,
mlflow_eval_result_adapter, and prompt_registry_adapter — was independently
duplicated across all three, with `mlflow_eval_result_adapter.py` disagreeing
on the default (`http://host.docker.internal:5000`, a vestige of a retired
docker-compose dev setup — no docker-compose.yml exists in this repo anymore).
`http://localhost:5000` is the canonical default: it already matches
`.env.example` and the other two adapters.
"""

import os

_DEFAULT_TRACKING_URI = "http://localhost:5000"


def get_mlflow_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)
