"""Shared kubeconfig bootstrap for every adapter that talks to the
Kubernetes API directly (kserve_adapter, openchoreo_inference_adapter,
openchoreo_promotion_adapter) — was duplicated verbatim in all three.
"""

from kubernetes import config


def load_kube_config_once() -> None:
    """In-cluster once this runs as a pod on worker1 (see
    infra/openchoreo/namespaces/default/projects/platform/components/
    orchestration-api/workload-orchestration-api.yaml); ConfigException means
    it's not running in a pod (local dev via `make run-orchestration-api`),
    so fall back to ~/.kube/config."""
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
