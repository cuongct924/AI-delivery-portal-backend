"""The real (non-mock) branch of factory.py's get_inference_adapter() — the
standard model-serving golden path's only backend now that Inference has
fully migrated off KServe-direct. Patches an OpenChoreo `Workload` instead of
an `InferenceService` directly; OpenChoreo's controller-manager renders the
real serving.kserve.io/v1beta1 InferenceService via ClusterComponentType
"proxy/inference-service"
(infra/openchoreo/platform-shared/component-types/clustercomponenttype-inference-service.yaml). Same
`CustomObjectsApi` convention as adapters/delivery/gpu_inference_adapter.py (GPU-serving-only
now), different CRD. Verified for real against the k3d cluster; the bugs it
surfaced (an RBAC fix in clusterrole-dataplane-kserve.yaml, plus two more) are
in the class docstring.

**Known, real, unresolved gap**: KServe's storage-initializer can't load this
repo's `models:/<name>/<version>` storage_uri convention (only gs://, s3://,
file://, http(s)://, hf:// are supported), and docker-compose.yml's mlflow
service serves artifacts behind its own mlflow-artifacts:/ proxy — so deployed
InferenceServices never actually load a model. Pre-existing in every models:/
callsite (routers/models.py, adapters/delivery/deploy_strategies.py); fixing needs a
real decision (MLflow artifact store on MinIO, or resolve model_uri first) out
of this adapter's scope.
"""

from collections.abc import Mapping
from typing import Final, cast

from kubernetes import client
from kubernetes.client.exceptions import ApiException

from adapters.delivery._kube_client import load_kube_config_once
from adapters.delivery.interfaces import DeployStatus, IInferenceAdapter

GROUP: Final[str] = "openchoreo.dev"
VERSION: Final[str] = "v1alpha1"
WORKLOAD_PLURAL: Final[str] = "workloads"

KSERVE_GROUP: Final[str] = "serving.kserve.io"
KSERVE_VERSION: Final[str] = "v1beta1"
INFERENCESERVICE_PLURAL: Final[str] = "inferenceservices"


class OpenChoreoInferenceAdapter(IInferenceAdapter):
    """Scoped to the one real serving Component (`serving`, under the
    `telco-fraud-detection` Project) — `deploy_model`'s `name`/`version` become
    the storageUri patched onto that fixed Workload rather than selecting/
    creating a Component per model. A second real model would need dynamic
    Component/Workload provisioning this adapter doesn't do — document as a gap.

    Two real, cluster-verified bugs surfaced by the rename to
    `telco-fraud-detection` (fixed as direct kubectl patches, not yet in
    GitOps-applied Helm values):
    1. `inferenceservice-config`'s `ingress.domainTemplate` hyphenated Name and
       Namespace into a single DNS label that overflowed the 63-char RFC 1035
       cap for any long auto-generated dataplane namespace. Fixed by separating
       them with dots and restarting kserve-controller-manager.
    2. A second, different 63-*byte* limit on the raw-mode predictor Service's
       label value — naming-sensitive: fixed by shortening the Component to
       `serving` (the Project name doesn't participate in this label).
    """

    def __init__(
        self,
        namespace: str = "default",
        project: str = "telco-fraud-detection",
        component: str = "serving",
        environment: str = "development",
    ):
        load_kube_config_once()
        self.namespace = namespace
        self.project = project
        self.component = component
        self.environment = environment
        self.api = client.CustomObjectsApi()

    def deploy_model(
        self,
        name: str,
        version: str,
        model_uri: str,
        traffic_fields: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        del name, version  # this Workload serves whichever model_uri it's patched to
        if traffic_fields and traffic_fields.get("canaryTrafficPercent") != 100:
            # ClusterComponentType "proxy/inference-service" has no canary
            # slot in environmentConfigs — this Workload has one
            # `container.image` field, no revision-split concept, so a
            # genuine PARTIAL split has no honest way to render. A 100%
            # split is exactly what deploy_strategies.py's rollback path
            # always sends (InstantStrategy via TrafficSplitStrategy(100)),
            # so treating it as a plain deploy is what makes rollback work.
            raise NotImplementedError(
                "OpenChoreoInferenceAdapter.deploy_model: a PARTIAL traffic split "
                f"({traffic_fields!r}) isn't wired into the InferenceService "
                "ClusterComponentType template yet — only a 100% cutover is supported"
            )
        workload_name = f"{self.component}-workload"
        patch = {"spec": {"container": {"image": model_uri}}}
        result = self.api.patch_namespaced_custom_object(
            GROUP, VERSION, self.namespace, WORKLOAD_PLURAL, workload_name, patch
        )
        return cast(dict[str, object], result)

    def get_inference_status(self, name: str) -> dict[str, object]:
        del name  # same single-Workload scoping as deploy_model
        # The real InferenceService lives in an auto-generated dataplane
        # namespace that can't be derived from `self.project`/`self.component`
        # — only discovered by listing across all namespaces, filtered by the
        # labels renderedrelease-controller actually sets.
        items = cast(
            dict[str, object],
            self.api.list_cluster_custom_object(
                KSERVE_GROUP,
                KSERVE_VERSION,
                INFERENCESERVICE_PLURAL,
                label_selector=(
                    f"openchoreo.dev/component={self.component},"
                    f"openchoreo.dev/environment={self.environment}"
                ),
            ),
        )
        matches = cast(list[dict[str, object]], items.get("items") or [])
        if not matches:
            raise ApiException(
                status=404,
                reason=(
                    f"No InferenceService found for component={self.component!r}, "
                    f"environment={self.environment!r} — has autoDeploy reconciled yet?"
                ),
            )
        return matches[0]

    def get_deploy_status(self, name: str) -> DeployStatus:
        try:
            status = self.get_inference_status(name)
        except ApiException as exc:
            if exc.status == 404:
                return DeployStatus(
                    deployed=False, ready=False, live_version=None, traffic_percent=None
                )
            raise

        spec = cast(dict[str, object], status.get("spec", {}))
        predictor = cast(dict[str, object], spec.get("predictor", {}))
        traffic_percent = cast(int | None, predictor.get("canaryTrafficPercent"))

        # No per-deploy version label exists here (unlike GpuKServeInferenceAdapter) —
        # the ClusterComponentType passes through whatever labels are on the
        # Workload, it doesn't set one per deploy. Parse the version instead
        # out of the "models:/<name>/<version>" storageUri deploy_model()
        # patched in — None if it isn't that shorthand.
        model_spec = cast(dict[str, object], predictor.get("model", {}))
        storage_uri = cast(str | None, model_spec.get("storageUri"))
        live_version = None
        if storage_uri is not None and storage_uri.startswith("models:/"):
            parts = storage_uri.removeprefix("models:/").rsplit("/", 1)
            if len(parts) == 2:
                live_version = parts[1]

        conditions = cast(
            list[dict[str, object]],
            cast(dict[str, object], status.get("status", {})).get("conditions", []),
        )
        ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)

        return DeployStatus(
            deployed=True, ready=ready, live_version=live_version, traffic_percent=traffic_percent
        )

    def predict(self, name: str, payload: dict[str, object]) -> dict[str, object]:
        del name, payload
        raise NotImplementedError(
            "Call the InferenceService's HTTP endpoint directly "
            "(get the URL from get_inference_status) instead of going through this adapter"
        )
