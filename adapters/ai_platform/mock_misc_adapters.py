"""Merges 3 trivial in-memory mocks with no divergent logic worth their own
files: MockEvalResultAdapter, MockNotebookAdapter, MockHuggingFaceHubAdapter.

Deliberately excludes mock_inference_adapter.py, mock_promotion_adapter.py,
mock_model_registry_adapter.py, mock_workflow_adapter.py — each of those
carries meaningful logic (real-exception-shape parity, a state machine
mirroring a real adapter's promotion/rollback contract, a gate-passing
metrics table, or a genuine phase-transition simulation) and stays standalone.
"""

import uuid
from datetime import datetime
from typing import Any

from adapters.ai_platform.interfaces import (
    HuggingFaceModelInfo,
    IEvalResultAdapter,
    IHuggingFaceHubAdapter,
    INotebookAdapter,
    NotebookDeletion,
    NotebookStatus,
)

# A few real, commonly-demoed HuggingFace ids with their actual published
# architecture — lets a demo/test pick a recognizable name and get realistic
# numbers back.
_KNOWN_HUGGINGFACE_MODELS: dict[str, HuggingFaceModelInfo] = {
    "meta-llama/Llama-3.1-8B-Instruct": HuggingFaceModelInfo(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        exists=True,
        is_gated=True,
        param_count_billion=8.03,
        max_context_length=131072,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        license="llama3.1",
    ),
    "mistralai/Mistral-7B-Instruct-v0.3": HuggingFaceModelInfo(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        exists=True,
        is_gated=False,
        param_count_billion=7.25,
        max_context_length=32768,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        license="apache-2.0",
    ),
    "does-not-exist/not-a-real-model": HuggingFaceModelInfo(
        model_id="does-not-exist/not-a-real-model",
        exists=False,
        is_gated=False,
        param_count_billion=None,
        max_context_length=None,
        num_layers=None,
        hidden_size=None,
        num_attention_heads=None,
        num_key_value_heads=None,
        license=None,
    ),
}


class MockEvalResultAdapter(IEvalResultAdapter):
    """In-memory Mock implementation of IEvalResultAdapter — used when
    USE_MOCK_ADAPTERS=true (default for local dev and CI)."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def log_judge_result(
        self,
        kind: str,
        name: str,
        version: str,
        judge_result: object,
        passed: bool,
    ) -> None:
        self._records.append(
            {
                "kind": kind,
                "name": name,
                "version": version,
                "judge_result": judge_result,
                "passed": passed,
                "timestamp": datetime.now(),
            }
        )

    def get_last_failure_at(self, kind: str, name: str) -> datetime | None:
        failures = [
            r["timestamp"]
            for r in self._records
            if r["kind"] == kind and r["name"] == name and r["passed"] is False
        ]
        if not failures:
            return None
        return max(failures)


class MockNotebookAdapter(INotebookAdapter):
    """Mock adapter for INotebookAdapter — in-memory stand-in for JupyterHub.
    Keeps the full resource profile so the Portal's provisioning form and a
    future management view can be exercised without a real spawner."""

    def __init__(self) -> None:
        self._notebooks: dict[str, NotebookStatus] = {}

    def create_notebook(
        self,
        environment: str,
        ram_gb: int,
        gpu_type: str | None = None,
        *,
        cpu_cores: int = 2,
        gpu_count: int = 1,
        storage_gb: int = 20,
        idle_timeout_minutes: int = 60,
    ) -> NotebookStatus:
        notebook_id = f"nb-{uuid.uuid4().hex[:8]}"
        now = datetime.utcnow().isoformat()
        status: NotebookStatus = {
            "notebook_id": notebook_id,
            "url": f"http://mock-jupyterhub.local/user/{notebook_id}",
            "active": True,
            "environment": environment,
            "cpu_cores": cpu_cores,
            "ram_gb": ram_gb,
            "gpu_type": gpu_type,
            # A CPU-only notebook has no GPU to count.
            "gpu_count": gpu_count if gpu_type else 0,
            "storage_gb": storage_gb,
            "idle_timeout_minutes": idle_timeout_minutes,
            "created_at": now,
            "last_activity_at": now,
        }
        self._notebooks[notebook_id] = status
        return status

    def get_notebook_status(self, notebook_id: str) -> NotebookStatus:
        status = self._notebooks.get(notebook_id)
        if status is None:
            raise ValueError(f"Notebook {notebook_id} does not exist")
        return status

    def list_notebooks(self) -> list[NotebookStatus]:
        return list(self._notebooks.values())

    def start_notebook(self, notebook_id: str) -> NotebookStatus:
        status = self.get_notebook_status(notebook_id)
        status["active"] = True
        status["url"] = f"http://mock-jupyterhub.local/user/{notebook_id}"
        status["last_activity_at"] = datetime.utcnow().isoformat()
        return status

    def stop_notebook(self, notebook_id: str) -> NotebookStatus:
        status = self.get_notebook_status(notebook_id)
        status["active"] = False
        status["url"] = None
        return status

    def delete_notebook(self, notebook_id: str) -> NotebookDeletion:
        if notebook_id not in self._notebooks:
            raise ValueError(f"Notebook {notebook_id} does not exist")
        del self._notebooks[notebook_id]
        return {"notebook_id": notebook_id, "deleted": True}


class MockHuggingFaceHubAdapter(IHuggingFaceHubAdapter):
    """Mock IHuggingFaceHubAdapter — no network calls, canned metadata for a
    handful of well-known model ids plus deterministic synthetic data for
    anything else, so the wizard's Model Source step and GPU sizing estimator
    work in CI/local dev without hitting the real Hub."""

    def get_model_info(self, model_id: str) -> HuggingFaceModelInfo:
        known = _KNOWN_HUGGINGFACE_MODELS.get(model_id)
        if known is not None:
            return known
        # Anything else: treat as a real, ungated 7B-class model for testing.
        return HuggingFaceModelInfo(
            model_id=model_id,
            exists=True,
            is_gated=False,
            param_count_billion=7.0,
            max_context_length=8192,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            license=None,
        )
