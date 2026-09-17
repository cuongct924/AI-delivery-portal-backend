"""Adapter for OpenChoreo's generic Workflow/WorkflowRun REST API — the
"openchoreo" branch of factory.py's get_workflow_adapter(); since Golden Paths
#1/#3 cut over, the only real workflow backend (the Argo Server adapter was
removed — see docs/openchoreo-workflow-migration-plan.md).

REST shape confirmed two ways (per docs/openchoreo-migration-next-steps.md
Phase 2): (1) read against openchoreo-workflows-backend's
GenericWorkflowService.ts in the backstage-plugins repo (POST/GET
/api/v1/namespaces/{ns}/workflowruns), and (2) a real WorkflowRun on the local
k3d cluster to see the actual response shape.

The live trigger also confirmed something the frontend code doesn't need: a
WorkflowRun's ClusterWorkflow still renders a real argoproj.io Workflow, just
in an auto-created `workflows-<namespace>` execution namespace. WorkflowRun's
own status is all this adapter needs — it never has to know that detail.
"""

import os
import time
from typing import Final, TypedDict

import httpx

from adapters.interfaces import IWorkflowAdapter, WorkflowStatus, WorkflowStepTiming

OPENCHOREO_API_URL: Final[str] = os.getenv(
    "OPENCHOREO_API_URL", "http://api.openchoreo.localhost:8080/api/v1"
)
THUNDER_URL: Final[str] = os.getenv("THUNDER_URL", "http://thunder.openchoreo.localhost:8080")
# Reuses the pre-existing "openchoreo-system-app" client (provisioned by
# thunder-bootstrap's 55-system-app.sh) — Thunder's /applications admin API
# 401s for every credential in dev, so registering a new client isn't
# possible without real Thunder admin access. Swap once a dedicated client
# exists; see infra/openchoreo/platform/authzrolebinding-workflow-trigger.yaml.
THUNDER_CLIENT_ID: Final[str] = os.getenv("OPENCHOREO_CLIENT_ID", "openchoreo-system-app")
THUNDER_CLIENT_SECRET: Final[str] = os.getenv(
    "OPENCHOREO_CLIENT_SECRET", "openchoreo-system-app-secret"
)

_cached_token: str | None = None
_cached_expiry: float = 0.0


def _get_access_token() -> str:
    """Same cache/refresh shape as thunder_client.py's get_access_token()
    (agents/mcp-servers/*) — not imported from there since adapters/ can't
    depend on agents/, and this is the only Thunder call this module makes."""
    global _cached_token, _cached_expiry
    now = time.monotonic()
    if _cached_token is not None and now < _cached_expiry:
        return _cached_token

    response = httpx.post(
        f"{THUNDER_URL}/oauth2/token",
        data={
            "grant_type": "client_credentials",
            "client_id": THUNDER_CLIENT_ID,
            "client_secret": THUNDER_CLIENT_SECRET,
        },
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    _cached_expiry = now + max(payload["expires_in"] - 10, 0)
    token: str = payload["access_token"]
    _cached_token = token
    return token


def _hyphen_to_camel(key: str) -> str:
    """ "hidden-size" -> "hiddenSize" — CEL identifiers can't contain hyphens,
    but trigger_workflow()'s callers build `parameters` with hyphenated
    argument names. Translating here keeps IWorkflowAdapter's contract
    backend-agnostic."""
    head, *rest = key.split("-")
    return head + "".join(word.capitalize() for word in rest)


class WorkflowSummary(TypedDict):
    name: str | None
    phase: str | None
    startedAt: str | None


def _derive_phase_and_message(conditions: list[dict[str, object]]) -> tuple[str, str | None]:
    """Maps WorkflowRun's status.conditions to the shared phase vocabulary
    ("Succeeded"/"Failed"/"Running"/"Pending") so routers/models.py's
    `phase in ("Succeeded", "Failed")` check works unchanged across backends.
    Priority order mirrors GenericWorkflowService.ts's deriveWorkflowRunStatus():
    WorkloadUpdated (component-build only) checked first.
    """
    by_type = {c["type"]: c for c in conditions if isinstance(c, dict) and "type" in c}

    def _true(condition_type: str) -> dict[str, object] | None:
        condition = by_type.get(condition_type)
        return condition if condition is not None and condition.get("status") == "True" else None

    for condition_type, phase in (
        ("WorkloadUpdated", "Succeeded"),
        ("WorkflowFailed", "Failed"),
        ("WorkflowSucceeded", "Succeeded"),
        ("WorkflowRunning", "Running"),
    ):
        condition = _true(condition_type)
        if condition is not None:
            message = condition.get("message")
            return phase, str(message) if message is not None else None
    return "Pending", None


class OpenChoreoWorkflowAdapter(IWorkflowAdapter):
    def __init__(self, base_url: str | None = None, namespace: str = "default"):
        self.base_url = base_url or OPENCHOREO_API_URL
        self.namespace = namespace

    def trigger_workflow(self, template_name: str, parameters: dict[str, str]) -> dict[str, object]:
        run_name = f"{template_name}-{int(time.time() * 1000)}"
        body = {
            "metadata": {"name": run_name},
            "spec": {
                "workflow": {
                    "kind": "ClusterWorkflow",
                    "name": template_name,
                    "parameters": {_hyphen_to_camel(k): v for k, v in parameters.items()},
                }
            },
        }
        response = httpx.post(
            f"{self.base_url}/namespaces/{self.namespace}/workflowruns",
            json=body,
            headers=self._headers(),
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def get_workflow_status(self, workflow_name: str) -> WorkflowStatus:
        response = httpx.get(
            f"{self.base_url}/namespaces/{self.namespace}/workflowruns/{workflow_name}",
            headers=self._headers(),
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        status = data.get("status", {})
        phase, message = _derive_phase_and_message(status.get("conditions", []))
        return {
            "name": workflow_name,
            "phase": phase,
            "message": message,
            "started_at": status.get("startedAt"),
            "finished_at": status.get("completedAt"),
            "steps": self._extract_step_timings(status),
        }

    def list_workflows(self) -> list[WorkflowSummary]:
        # Convenience method, not part of IWorkflowAdapter — same
        # list-of-summaries shape the routers expect from the real backend.
        response = httpx.get(
            f"{self.base_url}/namespaces/{self.namespace}/workflowruns",
            params={"limit": 100},
            headers=self._headers(),
            timeout=10,
        )
        response.raise_for_status()
        items = response.json().get("items") or []
        result: list[WorkflowSummary] = []
        for item in items:
            status = item.get("status", {})
            phase, _ = _derive_phase_and_message(status.get("conditions", []))
            result.append(
                {
                    "name": item.get("metadata", {}).get("name"),
                    "phase": phase,
                    "startedAt": status.get("startedAt"),
                }
            )
        return result

    def create_cron_workflow(
        self, name: str, schedule: str, workflow_template_name: str, parameters: dict[str, str]
    ) -> dict[str, object]:
        """Not implemented — openchoreo-api has no CronWorkflow-equivalent;
        the intended replacement is a per-model `Component`
        (ClusterComponentType "cronjob/scheduled-task", see
        infra/openchoreo/telco-fraud-detection/component-training.yaml), not a
        workflow trigger. The `telco-fraud-detection-training` Component/
        Workload is only applied for CI wiring, never exercised end to end.

        Blocker: scheduled-task's per-run `schedule` lives under
        `environmentConfigs` (supplied per-environment via a ReleaseBinding),
        not settable on Component.spec directly — confirmed by a
        `kubectl apply --dry-run=server` rejection. Implementing this needs a
        ReleaseBinding-creation call this adapter doesn't make yet.
        """
        raise NotImplementedError(
            "OpenChoreoWorkflowAdapter.create_cron_workflow: no CronWorkflow-equivalent "
            "in openchoreo-api's REST API yet, and the scheduled-task ClusterComponentType's "
            "per-instance schedule override isn't understood — see this method's docstring"
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {_get_access_token()}"}

    @staticmethod
    def _extract_step_timings(status: dict[str, object]) -> list[WorkflowStepTiming]:
        tasks = status.get("tasks")
        if not isinstance(tasks, list):
            return []
        steps: list[WorkflowStepTiming] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            steps.append(
                {
                    "name": str(task.get("name") or "unknown"),
                    "phase": task.get("phase"),
                    "started_at": task.get("startedAt"),
                    "finished_at": task.get("completedAt"),
                }
            )
        return steps
