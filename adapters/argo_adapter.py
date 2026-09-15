"""Adapter for Argo Workflows — triggers and tracks workflows via the Argo Server REST API.

Used for Golden Path #1 (Train → Track → Register): triggers the WorkflowTemplate
defined in infra/argo-workflows/, and tracks status to report back via Portal/Agent.
"""

import os
from typing import TypedDict

import httpx

from adapters.interfaces import IWorkflowAdapter, WorkflowStatus, WorkflowStepTiming


class WorkflowSummary(TypedDict):
    name: str | None
    phase: str | None
    startedAt: str | None


# TODO: expose via orchestration-api for the `orchestration:trigger-training`
# Custom Scaffolder Action (Golden Path #1) to call.
class ArgoAdapter(IWorkflowAdapter):
    def __init__(self, base_url: str | None = None, namespace: str = "default"):
        self.base_url = base_url or os.getenv("ARGO_SERVER_URL", "http://localhost:2746")
        self.namespace = namespace

    def trigger_workflow(self, template_name: str, parameters: dict[str, str]) -> dict[str, object]:
        payload = {
            "resourceKind": "WorkflowTemplate",
            "resourceName": template_name,
            "submitOptions": {"parameters": [f"{k}={v}" for k, v in parameters.items()]},
        }
        response = httpx.post(
            f"{self.base_url}/api/v1/workflows/{self.namespace}/submit",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def get_workflow_status(self, workflow_name: str) -> WorkflowStatus:
        response = httpx.get(
            f"{self.base_url}/api/v1/workflows/{self.namespace}/{workflow_name}",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        status = data.get("status", {})
        return {
            "name": workflow_name,
            "phase": status.get("phase"),
            # Surfaces the failure reason (e.g. pod OOMKilled) when phase is Failed/Error.
            "message": status.get("message"),
            "started_at": status.get("startedAt"),
            "finished_at": status.get("finishedAt"),
            "steps": self._extract_step_timings(status),
        }

    @staticmethod
    def _extract_step_timings(status: dict[str, object]) -> list[WorkflowStepTiming]:
        """`status.nodes` is Argo's full node tree (Pod nodes plus
        structural Steps/DAG/Retry nodes) — only `type == "Pod"` nodes are
        real executed steps, which is what RQ1's step-duration metric
        (dora_metrics.py) needs."""
        nodes = status.get("nodes")
        if not isinstance(nodes, dict):
            return []
        steps: list[WorkflowStepTiming] = []
        for node in nodes.values():
            if not isinstance(node, dict) or node.get("type") != "Pod":
                continue
            steps.append(
                {
                    "name": str(node.get("displayName") or node.get("name") or "unknown"),
                    "phase": node.get("phase"),
                    "started_at": node.get("startedAt"),
                    "finished_at": node.get("finishedAt"),
                }
            )
        return steps

    def create_cron_workflow(
        self, name: str, schedule: str, workflow_template_name: str, parameters: dict[str, str]
    ) -> dict[str, object]:
        """Creates (or replaces) a CronWorkflow — same CRD family as
        WorkflowTemplate, no new infra. Used by "Setup Model Monitoring" to
        register a periodic drift-check job; `name` is deterministic (1
        CronWorkflow per model name) so re-running Setup updates the
        existing schedule/threshold instead of creating a duplicate.

        Not part of IWorkflowAdapter — same precedent as list_workflows().
        """
        body = {
            "cronWorkflow": {
                "apiVersion": "argoproj.io/v1alpha1",
                "kind": "CronWorkflow",
                "metadata": {"name": name},
                "spec": {
                    "schedule": schedule,
                    "concurrencyPolicy": "Replace",
                    "workflowSpec": {
                        "workflowTemplateRef": {"name": workflow_template_name},
                        "arguments": {
                            "parameters": [{"name": k, "value": v} for k, v in parameters.items()]
                        },
                    },
                },
            }
        }
        response = httpx.post(
            f"{self.base_url}/api/v1/cron-workflows/{self.namespace}",
            json=body,
            timeout=10,
        )
        if response.status_code == 409:
            # Already exists — re-running "Setup Model Monitoring" for the
            # same model updates its schedule/threshold in place.
            response = httpx.put(
                f"{self.base_url}/api/v1/cron-workflows/{self.namespace}/{name}",
                json=body,
                timeout=10,
            )
        response.raise_for_status()
        return response.json()

    def list_workflows(self) -> list[WorkflowSummary]:
        # Convenience method, not part of IWorkflowAdapter — same precedent
        # as QdrantAdapter.ensure_collection() in vector_db_adapter.py.
        response = httpx.get(
            f"{self.base_url}/api/v1/workflows/{self.namespace}",
            timeout=10,
        )
        response.raise_for_status()
        items = response.json().get("items") or []
        return [
            {
                "name": item.get("metadata", {}).get("name"),
                "phase": item.get("status", {}).get("phase"),
                "startedAt": item.get("status", {}).get("startedAt"),
            }
            for item in items
        ]
