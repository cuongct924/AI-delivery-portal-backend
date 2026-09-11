"""Mock adapter for IWorkflowAdapter — stands in for Argo Workflows when no
`kind` cluster/Argo Server is up, so Golden Path #1 (Train -> Track ->
Register) and "Setup Model Monitoring" can be demoed end to end.

A triggered workflow reports phase="Running" on its first status check and
phase="Succeeded" from the second check onward — enough of a fake delay for
a demo to see a real state transition without waiting on an actual run.

Enable via `USE_MOCK_ADAPTERS=true` (adapters/factory.py).
"""

import uuid
from datetime import UTC, datetime
from typing import TypedDict

from adapters.interfaces import IWorkflowAdapter, WorkflowStatus
from adapters.mock_model_registry_adapter import MockModelRegistryAdapter


class WorkflowSummary(TypedDict):
    name: str | None
    phase: str | None
    startedAt: str | None


class _WorkflowRecord(TypedDict):
    phase: str
    startedAt: str
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
            "status_checks": 0,
        }
        self._maybe_register_model(parameters)
        return {"metadata": {"name": name}, "status": {"phase": "Running"}}

    def get_workflow_status(self, workflow_name: str) -> WorkflowStatus:
        record = self._workflows.get(workflow_name)
        if record is None:
            return {"name": workflow_name, "phase": None, "message": "workflow not found"}
        record["status_checks"] += 1
        if record["status_checks"] > 1:
            record["phase"] = "Succeeded"
        return {"name": workflow_name, "phase": record["phase"], "message": None}

    def create_cron_workflow(
        self, name: str, schedule: str, workflow_template_name: str, parameters: dict[str, str]
    ) -> dict[str, object]:
        """Convenience method, not part of IWorkflowAdapter — mirrors
        ArgoAdapter.create_cron_workflow() so factory.py can hand either one
        to routers/monitoring.py without it noticing."""
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
        """Convenience method, not part of IWorkflowAdapter — same precedent
        as ArgoAdapter.list_workflows()."""
        return [
            {"name": name, "phase": record["phase"], "startedAt": record["startedAt"]}
            for name, record in self._workflows.items()
        ]

    def _maybe_register_model(self, parameters: dict[str, str]) -> None:
        """Mirrors the callback infra/argo-workflows/train-register-template.yaml's
        register-step makes for real (and scripts/local-demo/fake_argo.py
        makes for the real-training local demo) — without this, a Golden
        Path #1 run under USE_MOCK_ADAPTERS=true reports its workflow
        "Succeeded" but never registers anything, so the very next call
        (GET /models/{name}/latest-version) 500s against an empty registry.
        No-ops when the model registry isn't ALSO mocked (factory.py only
        passes one in that case) or when `parameters` isn't a training
        request (e.g. "Setup Model Monitoring"'s create_cron_workflow uses
        a different parameter shape entirely, never reaches here).
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
