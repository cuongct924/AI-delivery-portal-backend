"""Mock adapter for IWorkflowAdapter — stands in for the real workflow
backend when no OpenChoreo cluster is up.

A triggered workflow reports phase="Running" on its first status check and
phase="Succeeded" from the second onward, to demo a real state transition.
"""

import uuid
from datetime import UTC, datetime
from typing import TypedDict

from adapters.interfaces import IWorkflowAdapter, WorkflowStatus, WorkflowStepTiming
from adapters.mock_model_registry_adapter import MockModelRegistryAdapter


class WorkflowSummary(TypedDict):
    name: str | None
    phase: str | None
    startedAt: str | None


class _WorkflowRecord(TypedDict):
    phase: str
    startedAt: str
    finishedAt: str | None
    status_checks: int


class MockWorkflowAdapter(IWorkflowAdapter):
    def __init__(self, model_registry: MockModelRegistryAdapter | None = None) -> None:
        self._workflows: dict[str, _WorkflowRecord] = {}
        self._cron_workflows: dict[str, dict[str, object]] = {}
        # Only wired by factory.py when the model registry is ALSO mocked —
        # see _maybe_register_model()'s docstring.
        self._model_registry = model_registry

    def trigger_workflow(self, template_name: str, parameters: dict[str, str]) -> dict[str, object]:
        name = f"{template_name}-{uuid.uuid4().hex[:8]}"
        self._workflows[name] = {
            "phase": "Running",
            "startedAt": datetime.now(UTC).isoformat(),
            "finishedAt": None,
            "status_checks": 0,
        }
        self._maybe_register_model(parameters)
        return {"metadata": {"name": name}, "status": {"phase": "Running"}}

    def get_workflow_status(self, workflow_name: str) -> WorkflowStatus:
        record = self._workflows.get(workflow_name)
        if record is None:
            return {
                "name": workflow_name,
                "phase": None,
                "message": "workflow not found",
                "started_at": None,
                "finished_at": None,
                "steps": [],
            }
        record["status_checks"] += 1
        if record["status_checks"] > 1:
            record["phase"] = "Succeeded"
            if record["finishedAt"] is None:
                record["finishedAt"] = datetime.now(UTC).isoformat()
        steps: list[WorkflowStepTiming] = [
            {
                "name": "train",
                "phase": record["phase"],
                "started_at": record["startedAt"],
                "finished_at": record["finishedAt"],
            }
        ]
        return {
            "name": workflow_name,
            "phase": record["phase"],
            "message": None,
            "started_at": record["startedAt"],
            "finished_at": record["finishedAt"],
            "steps": steps,
        }

    def create_cron_workflow(
        self, name: str, schedule: str, workflow_template_name: str, parameters: dict[str, str]
    ) -> dict[str, object]:
        """Convenience method, not part of IWorkflowAdapter — returns an
        Argo-style CronWorkflow resource. The mock is the only backend that
        supports scheduled monitoring today: OpenChoreoWorkflowAdapter has
        no CronWorkflow equivalent and raises NotImplementedError."""
        cron_workflow = {
            "metadata": {"name": name},
            "spec": {
                "schedule": schedule,
                "workflowSpec": {
                    "workflowTemplateRef": {"name": workflow_template_name},
                    "arguments": {
                        "parameters": [{"name": k, "value": v} for k, v in parameters.items()]
                    },
                },
            },
        }
        self._cron_workflows[name] = cron_workflow
        return {"cronWorkflow": cron_workflow}

    def list_workflows(self) -> list[WorkflowSummary]:
        """Convenience method, not part of IWorkflowAdapter — same
        list-of-summaries shape the routers expect from the real backend."""
        return [
            {"name": name, "phase": record["phase"], "startedAt": record["startedAt"]}
            for name, record in self._workflows.items()
        ]

    def _maybe_register_model(self, parameters: dict[str, str]) -> None:
        """Mirrors the register-step callback a real training run makes —
        without it, a pure-mock run 500s on the next latest-version lookup.
        No-ops when the registry isn't also mocked, or `parameters` isn't a
        training request.
        """
        if self._model_registry is None:
            return
        model_name = parameters.get("model-name")
        task_type = parameters.get("task-type")
        if not model_name or not task_type:
            return
        result = self._model_registry.register_model(model_name, "mock://artifact")
        self._model_registry.set_model_version_tag(
            model_name, result["version"], "task_type", task_type
        )
