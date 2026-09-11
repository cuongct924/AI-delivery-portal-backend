"""Fake Argo Server, standing in for the real kind cluster + Argo Workflows
during a local demo of Golden Path #1 (Train -> Track -> Register), for
every `architecture` (sklearn/mlp/lstm/nlp/cv) — not just classical ML.

Implements just the 2 endpoints adapters/argo_adapter.py (AI-delivery-portal)
calls:
  POST /api/v1/workflows/{namespace}/submit
  GET  /api/v1/workflows/{namespace}/{name}

On submit, it runs the *real* training image entrypoint
(infra/argo-workflows/training-image/train.py) as a subprocess, with the
same env vars infra/argo-workflows/train-register-template.yaml's
`train-step` container sets — then calls back into orchestration-api's
`POST /models/register` (not mlflow.register_model directly), the exact
same callback the template's `register-step` container makes. MLflow ends
up with a real run + a real registered model version; only the Argo/
Kubernetes layer is faked, and — unlike an earlier version of this script —
the training logic itself is never re-implemented here, so it can never
drift from what a real training pod actually does (metric names included).

Only one training run at a time: train.py writes its outputs to the fixed
paths /tmp/artifact-uri and /tmp/dataset-digest (same as the real
container, which gets a fresh /tmp per pod — this process doesn't), so
concurrent runs would clobber each other's output files.

Run: AI_DELIVERY_PORTAL/.venv/bin/python scripts/local-demo/fake_argo.py
Requires: mlflow (docker compose up -d mlflow) and orchestration-api
(make run-orchestration-api) both reachable — see MLFLOW_TRACKING_URI /
ORCHESTRATION_API_URL below to point elsewhere.
Listens on :2746 (ArgoAdapter's default ARGO_SERVER_URL).
"""

import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request

_TRAINING_IMAGE_DIR = Path(__file__).resolve().parents[2] / "infra/argo-workflows/training-image"
_ARTIFACT_URI_PATH = Path("/tmp/artifact-uri")  # noqa: S108 - matches train.py's own hardcoded path
_DATASET_DIGEST_PATH = Path("/tmp/dataset-digest")  # noqa: S108 - ditto

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
ORCHESTRATION_API_URL = os.getenv("ORCHESTRATION_API_URL", "http://localhost:8000")

# kebab-case (as sent by routers/models.py's trigger_training()) -> the env
# var name train.py's main() reads. Values not present in a given request's
# `params` fall back to _ENV_DEFAULTS below, mirroring
# train-register-template.yaml's own per-parameter `value: ""` defaults.
_ENV_KEY_MAP: dict[str, str] = {
    "dataset-uri": "DATASET_URI",
    "task-type": "TASK_TYPE",
    "target-column": "TARGET_COLUMN",
    "id-columns": "ID_COLUMNS",
    "architecture": "ARCHITECTURE",
    "algorithm": "ALGORITHM",
    "mode": "MODE",
    "base-model-uri": "BASE_MODEL_URI",
    "time-column": "TIME_COLUMN",
    "hidden-layers": "HIDDEN_LAYERS",
    "dropout": "DROPOUT",
    "sequence-length": "SEQUENCE_LENGTH",
    "num-layers": "NUM_LAYERS",
    "hidden-size": "HIDDEN_SIZE",
    "learning-rate": "LEARNING_RATE",
    "epochs": "EPOCHS",
    "batch-size": "BATCH_SIZE",
    "optimizer": "OPTIMIZER",
    "code-repo-url": "CODE_REPO_URL",
    "entrypoint-path": "ENTRYPOINT_PATH",
    "custom-config": "CUSTOM_CONFIG",
    "search-strategy": "SEARCH_STRATEGY",
    "num-trials": "NUM_TRIALS",
    "search-space-json": "SEARCH_SPACE_JSON",
    "objective-metric": "OBJECTIVE_METRIC",
    "objective-direction": "OBJECTIVE_DIRECTION",
    "text-column": "TEXT_COLUMN",
    "base-model-name": "BASE_MODEL_NAME",
}
_ENV_DEFAULTS: dict[str, str] = {
    "ARCHITECTURE": "sklearn",
    "MODE": "train",
    "CUSTOM_CONFIG": "{}",
    "SEARCH_STRATEGY": "fixed",
    "SEARCH_SPACE_JSON": "{}",
    "OBJECTIVE_DIRECTION": "maximize",
}

app = FastAPI()

# workflow_name -> {"phase": str, "message": str | None}
_WORKFLOWS: dict[str, dict[str, Any]] = {}
# Guards the fixed /tmp/artifact-uri /tmp/dataset-digest output paths — see
# module docstring.
_TRAIN_LOCK = threading.Lock()


def _register_model(
    model_name: str, task_type: str, artifact_uri: str, dataset_digest: str
) -> None:
    """Calls back into the real orchestration-api endpoint — the same
    callback infra/argo-workflows/train-register-template.yaml's
    register-step container makes — instead of registering directly via the
    mlflow SDK, so this demo actually exercises routers/models.py's
    register_model() (task_type tagging included), not a shortcut around it.
    """
    response = httpx.post(
        f"{ORCHESTRATION_API_URL}/models/register",
        json={
            "name": model_name,
            "artifact_uri": artifact_uri,
            "task_type": task_type,
            "dataset_version": dataset_digest or None,
        },
        timeout=30,
    )
    response.raise_for_status()


def _run_training(params: dict[str, str]) -> None:
    model_name = params["model-name"]
    task_type = params["task-type"]

    env = {**os.environ, **_ENV_DEFAULTS}
    for key, env_key in _ENV_KEY_MAP.items():
        env[env_key] = params.get(key, env.get(env_key, ""))
    env["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    env["PYTHONPATH"] = str(_TRAINING_IMAGE_DIR)

    with _TRAIN_LOCK:
        _ARTIFACT_URI_PATH.unlink(missing_ok=True)
        _DATASET_DIGEST_PATH.unlink(missing_ok=True)
        result = subprocess.run(  # noqa: S603 - fixed local script, not user input
            [sys.executable, str(_TRAINING_IMAGE_DIR / "train.py")],
            env=env,
            cwd=_TRAINING_IMAGE_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"train.py failed:\n{result.stderr[-4000:]}")
        artifact_uri = _ARTIFACT_URI_PATH.read_text().strip()
        dataset_digest = _DATASET_DIGEST_PATH.read_text().strip()

    _register_model(model_name, task_type, artifact_uri, dataset_digest)


def _train_in_background(workflow_name: str, params: dict[str, str]) -> None:
    try:
        _run_training(params)
        _WORKFLOWS[workflow_name] = {"phase": "Succeeded", "message": None}
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as workflow status
        _WORKFLOWS[workflow_name] = {"phase": "Failed", "message": str(exc)}


@app.post("/api/v1/workflows/{namespace}/submit")
async def submit(namespace: str, request: Request) -> dict:
    body = await request.json()
    raw_params: list[str] = body.get("submitOptions", {}).get("parameters", [])
    params = dict(p.split("=", 1) for p in raw_params)

    # Real Argo returns as soon as the Workflow object is created, well
    # before the pod finishes -- ArgoAdapter.trigger_workflow's own httpx
    # call has only a 10s timeout, so training (mlflow.sklearn.log_model's
    # first call alone can take longer, capturing the env/signature) must
    # run in the background and be polled via GET .../status, not inline.
    workflow_name = f"train-register-golden-path-{uuid.uuid4().hex[:8]}"
    _WORKFLOWS[workflow_name] = {"phase": "Running", "message": None}
    threading.Thread(target=_train_in_background, args=(workflow_name, params), daemon=True).start()

    return {"metadata": {"name": workflow_name}}


@app.get("/api/v1/workflows/{namespace}/{name}")
async def status(namespace: str, name: str) -> dict:
    entry = _WORKFLOWS.get(name, {"phase": "Failed", "message": "unknown workflow"})
    return {"status": entry}


if __name__ == "__main__":
    print(f"fake-argo listening on :2746, logging to {MLFLOW_TRACKING_URI}")
    print(f"registering models via {ORCHESTRATION_API_URL}/models/register")
    uvicorn.run(app, host="0.0.0.0", port=2746)
