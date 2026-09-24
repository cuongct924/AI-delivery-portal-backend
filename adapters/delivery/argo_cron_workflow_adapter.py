"""Argo CronWorkflow creator for scheduled monitoring — the real backend
behind `POST /setup-monitoring`'s `create_cron_workflow()` call.

Why Argo directly instead of OpenChoreo like every other Golden Path:
openchoreo-api has no CronWorkflow-equivalent (see
OpenChoreoWorkflowAdapter.create_cron_workflow, which raises
NotImplementedError), while the Argo controller is already in the cluster
running every training Workflow. So this adapter talks to the Kubernetes
API itself (CustomObjectsApi, same load_kube_config_once() bootstrap as the
KServe/OpenChoreo adapters) and creates a namespaced Argo CronWorkflow with
an inline workflowSpec — no WorkflowTemplate dependency to apply or drift.

Placement mirrors the training path: the CronWorkflow lives in
`workflows-default` and runs as `workflow-sa` (auto-provisioned there with
the outputs-reporting Role), pinned onto worker2 where the /mnt/data
hostPath resolves, using the same training-image:local that already ships
monitor_drift.py. In-cluster Service DNS everywhere (MLflow,
orchestration-api, MinIO) — never host.docker.internal, which is
unreachable from a pod.
"""

from typing import Any, Final, cast

from kubernetes import client
from kubernetes.client.rest import ApiException

from adapters.delivery._kube_client import load_kube_config_once

_CRON_GROUP: Final[str] = "argoproj.io"
_CRON_VERSION: Final[str] = "v1alpha1"
_CRON_PLURAL: Final[str] = "cronworkflows"

# Same execution namespace OpenChoreo's ClusterWorkflow mechanism renders
# training Workflows into — keeps every Golden Path run in one place.
_NAMESPACE: Final[str] = "workflows-default"
_SERVICE_ACCOUNT: Final[str] = "workflow-sa"
_IMAGE: Final[str] = "training-image:local"

# Hardcoded in-cluster DNS, not process env: these land in the *pod's*
# environment, and the API process itself may run on a dev host whose
# MLFLOW_TRACKING_URI points at localhost (unreachable from a pod).
_MLFLOW_TRACKING_URI: Final[str] = "http://mlflow.ai-platform-zone.svc.cluster.local:5000"
# Same Service DNS register-step uses to call back into the API — see
# train-register-cluster-template.yaml.
_ORCHESTRATION_API_URL: Final[str] = (
    "http://orchestration-api.dp-default-platform-development-8260e221.svc.cluster.local:8000"
)
_MINIO_ENDPOINT_URL: Final[str] = "http://minio.ai-platform-zone.svc.cluster.local:9000"
# Same MinIO dev creds used everywhere else in this repo, not a new secret.
_AWS_ACCESS_KEY_ID: Final[str] = "minioadmin"
_AWS_SECRET_ACCESS_KEY: Final[str] = "minioadmin"

# setup-monitoring's hyphenated parameter names → monitor_drift.py's env.
_PARAM_TO_ENV: Final[dict[str, str]] = {
    "model-name": "MODEL_NAME",
    "model-version": "MODEL_VERSION",
    "reference-data-uri": "REFERENCE_DATA_URI",
    "production-data-uri": "PRODUCTION_DATA_URI",
    "monitoring-type": "MONITORING_TYPE",
    "drift-threshold": "DRIFT_THRESHOLD",
    "ground-truth-data-uri": "GROUND_TRUTH_DATA_URI",
    "metric-names": "METRIC_NAMES",
    "metric-thresholds": "METRIC_THRESHOLDS",
    "min-metric-threshold": "MIN_METRIC_THRESHOLD",
    "on-drift-detected": "ON_DRIFT_DETECTED",
    "retrain-request-json": "RETRAIN_REQUEST_JSON",
    "failure-webhook-url": "FAILURE_WEBHOOK_URL",
}


class ArgoCronWorkflowAdapter:
    """Creates Argo CronWorkflows for recurring drift checks. Deliberately
    not an IWorkflowAdapter — triggering/polling one-shot training runs is
    OpenChoreo's job; this class only owns the scheduled path."""

    def __init__(self) -> None:
        load_kube_config_once()
        self._api = client.CustomObjectsApi()

    def create_cron_workflow(
        self, name: str, schedule: str, workflow_template_name: str, parameters: dict[str, str]
    ) -> dict[str, object]:
        """Creates (or refreshes) the named CronWorkflow. Re-running Setup
        for the same model updates the schedule/spec in place instead of
        duplicating it — matches the deterministic `monitor-{model}` naming
        routers/monitoring.py uses.

        `workflow_template_name` is accepted for signature parity with the
        mock and recorded as a label, but the spec is inline (no template
        to keep in sync).
        """
        del workflow_template_name
        body: dict[str, Any] = {
            "apiVersion": f"{_CRON_GROUP}/{_CRON_VERSION}",
            "kind": "CronWorkflow",
            "metadata": {
                "name": name,
                "labels": {
                    "app.kubernetes.io/managed-by": "orchestration-api",
                    "app.kubernetes.io/part-of": "setup-model-monitoring",
                },
            },
            "spec": {
                # Plural `schedules` — Argo >= 3.5 renamed the old singular
                # `schedule` string; the old name 422s.
                "schedules": [schedule],
                # A run that overruns its slot is skipped, not piled up.
                "concurrencyPolicy": "Forbid",
                "successfulJobsHistoryLimit": 3,
                "failedJobsHistoryLimit": 3,
                "workflowSpec": self._workflow_spec(parameters),
            },
        }
        try:
            created = self._api.create_namespaced_custom_object(
                _CRON_GROUP, _CRON_VERSION, _NAMESPACE, _CRON_PLURAL, body
            )
        except ApiException as e:
            if e.status != 409:
                raise
            current = cast(
                dict[str, Any],
                self._api.get_namespaced_custom_object(
                    _CRON_GROUP, _CRON_VERSION, _NAMESPACE, _CRON_PLURAL, name
                ),
            )
            current["spec"] = body["spec"]
            created = self._api.replace_namespaced_custom_object(
                _CRON_GROUP, _CRON_VERSION, _NAMESPACE, _CRON_PLURAL, name, current
            )
        return {"cronWorkflow": created}

    @staticmethod
    def _workflow_spec(parameters: dict[str, str]) -> dict[str, Any]:
        """Single-step spec running monitor_drift.py — mirrors
        infra/argo-workflows/monitor-drift-template.yaml's container (which
        stays as documentation of the pre-OpenChoreo wiring), with the
        service account, node pinning, and in-cluster endpoints fixed for
        the current topology."""
        env = [
            {"name": "MLFLOW_TRACKING_URI", "value": _MLFLOW_TRACKING_URI},
            {"name": "AWS_ACCESS_KEY_ID", "value": _AWS_ACCESS_KEY_ID},
            {"name": "AWS_SECRET_ACCESS_KEY", "value": _AWS_SECRET_ACCESS_KEY},
            {"name": "MLFLOW_S3_ENDPOINT_URL", "value": _MINIO_ENDPOINT_URL},
            {"name": "ORCHESTRATION_API_URL", "value": _ORCHESTRATION_API_URL},
        ]
        for param, value in parameters.items():
            env_name = _PARAM_TO_ENV.get(param)
            if env_name is not None:
                env.append({"name": env_name, "value": value})
        return {
            "entrypoint": "monitor",
            "serviceAccountName": _SERVICE_ACCOUNT,
            "volumes": [
                {"name": "dataset", "hostPath": {"path": "/mnt/data", "type": "Directory"}}
            ],
            "templates": [
                {
                    "name": "monitor",
                    "nodeSelector": {"plane.viettel.vn": "ai-platform-workflow"},
                    "tolerations": [
                        {
                            "key": "dedicated.viettel.vn",
                            "operator": "Equal",
                            "value": "ai-platform-workflow",
                            "effect": "NoSchedule",
                        }
                    ],
                    "container": {
                        "image": _IMAGE,
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["python", "monitor_drift.py"],
                        "env": env,
                        "volumeMounts": [
                            {"name": "dataset", "mountPath": "/mnt/data", "readOnly": True}
                        ],
                    },
                }
            ],
        }
