"""Adapter for the "openchoreo" branch of factory.py's get_kserve_adapter()
— patches an OpenChoreo `Workload` object rather than an `InferenceService`
directly, letting OpenChoreo's own controller-manager render+apply the
real serving.kserve.io/v1beta1 InferenceService (via
infra/openchoreo/platform/clustercomponenttype-inference-service.yaml's
ClusterComponentType "proxy/inference-service") — matches the same
`kubernetes.client.CustomObjectsApi` convention as adapters/kserve_adapter.py,
just against a different CRD.

Verified for real against the k3d cluster (`k3d-openchoreo-quick-start`):
applying infra/openchoreo/telco-fraud-detection/{component,workload}-serving.yaml
+ a ClusterAuthzRole fix (see infra/openchoreo/platform/
clusterrole-dataplane-kserve.yaml — OpenChoreo's own dataplane reconciler
agent had no RBAC for serving.kserve.io at all, added after KServe, a real
blocker this session actually hit) made the reconciler render and apply a
real InferenceService (originally `fraud-detection-serving-development-a1dd8967`,
confirmed via `kubectl get inferenceservice -n
dp-default-fraud-detecti-development-d0e7d311` — note the auto-generated
dataplane namespace name, confirming the migration doc's own warning that
it can't be assumed/derived, only discovered; the Project/Component were
later renamed to telco-fraud-detection/serving, which surfaced two more
real, since-fixed bugs — see this module's own class docstring below).

**Known, real, unresolved gap** (not fixed here — out of this phase's
scope, and not an OpenChoreo-specific problem): the InferenceService gets
created and its ClusterServingRuntime resolves correctly
(modelFormat=mlflow/version=2, protocolVersion=v2 — kserve-mlserver), but
KServe's storage-initializer cannot actually load the model —
`models:/<name>/<version>` (this repo's existing storage_uri convention,
shared with adapters/kserve_adapter.py's InstantStrategy/
prepare_deploy_manifest, both predating this file) is MLflow's own
registry-reference scheme, not one KServe's storage-initializer
understands (only gs://, s3://, file://, http(s)://, hf:// — confirmed via
a raw `kubectl apply` InferenceService with the exact same storageUri:
"Cannot recognize storage type for models:/..."). MLflow's own artifact
store here is also unreachable by any of those schemes as configured
(docker-compose.yml's mlflow service runs with no --artifacts-destination,
so artifacts sit behind MLflow's own mlflow-artifacts:/ proxy scheme, not
a real S3/GCS bucket). This is a pre-existing gap in every
model_uri = f"models:/{name}/{version}" callsite (routers/models.py,
adapters/deploy_strategies.py) — it just could never surface until this
session's Phase 3 put a real KServe control-plane in place to test
against. Fixing it needs a real decision (point MLflow's artifact store at
the existing MinIO service, or resolve model_uri to a real artifact
location before handing it to KServe) that's bigger than this adapter and
was not made this session.
"""

from collections.abc import Mapping
from typing import Final, cast

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from adapters.interfaces import IInferenceAdapter

GROUP: Final[str] = "openchoreo.dev"
VERSION: Final[str] = "v1alpha1"
WORKLOAD_PLURAL: Final[str] = "workloads"

KSERVE_GROUP: Final[str] = "serving.kserve.io"
KSERVE_VERSION: Final[str] = "v1beta1"
INFERENCESERVICE_PLURAL: Final[str] = "inferenceservices"


class OpenChoreoInferenceAdapter(IInferenceAdapter):
    """Scoped to a single, fixed Component/Workload — this repo has exactly
    one real serving Component (`serving`, under the `telco-fraud-detection`
    Project — see
    infra/openchoreo/telco-fraud-detection/{component,workload}-serving.yaml),
    so `deploy_model`'s `name`/`version` become the storageUri patched onto
    that one Workload rather than selecting/creating a Component per model
    name. Generalizing to "one Component per model" would need dynamic
    Component/Workload provisioning this adapter doesn't do — not
    attempted here; document as a gap if a second real model ever needs
    this.

    Two more real, cluster-verified bugs surfaced by the
    fraud-detection -> telco-fraud-detection rename, both since fixed live
    on the cluster (not yet reflected in any GitOps-applied Helm
    values — these are direct kubectl patches):

    1. `inferenceservice-config`'s `ingress.domainTemplate` defaulted to
       `"{{ .Name }}-{{ .Namespace }}.{{ .IngressDomain }}"` — hyphenating
       Name and Namespace into a single DNS label instead of separating
       them with a dot. Combined with OpenChoreo's long auto-generated
       dataplane namespace (`dp-default-<project>-<environment>-<hash>`),
       this label exceeded the 63-character RFC 1035 cap regardless of
       project/component naming (the fixed structural overhead alone
       already exceeds 63 for the "development" environment). Fixed by
       patching the template to `"{{ .Name }}.{{ .Namespace }}.{{
       .IngressDomain }}"` (three separate labels, each well under 63) and
       restarting kserve-controller-manager.
    2. With that fixed, a second, different 63-*byte* limit surfaced: the
       raw-mode predictor Service's `isvc.<name>-predictor` label value
       (a Kubernetes label value, not a DNS hostname — same numeric limit,
       different mechanism) still overflowed with `component-serving`
       naming. This one *is* naming-sensitive: fixed by shortening the
       Component to `serving` (was `telco-fraud-detection-serving`) — the
       Project name doesn't participate in this label at all, only
       `<component>-<environment>-<hash>` does.
    """

    def __init__(
        self,
        namespace: str = "default",
        project: str = "telco-fraud-detection",
        component: str = "serving",
        environment: str = "development",
    ):
        config.load_kube_config()
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
            # ClusterComponentType "proxy/inference-service" has no
            # canaryTrafficPercent-equivalent slot in its environmentConfigs
            # yet (infra/openchoreo/platform/clustercomponenttype-inference-service.yaml)
            # — this Workload has exactly one `container.image` field, no
            # "previous vs. new revision" split concept at all, so a
            # genuine PARTIAL split (1-99%, or 0% "staged dark") has no
            # honest way to render here. A 100% split is a different
            # case: it's requesting the exact same outcome as patching the
            # image directly (send everything to the new version) —
            # deploy_strategies.py's rollback path always sends exactly
            # this (adapters/deploy_strategies.py's InstantStrategy via
            # TrafficSplitStrategy(100)), so treating it as a plain deploy
            # is what makes rollback (and blue-green at 100%) actually
            # work against this adapter instead of always raising.
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

    def deploy_llm_model(
        self,
        name: str,
        version: str,
        huggingface_model_id: str,
        serving_runtime_name: str,
        gpu_count: int,
        vllm_quantization: str | None,
        max_context_length: int,
        traffic_fields: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Convenience method, not part of IInferenceAdapter — mirrors
        KServeAdapter.deploy_llm_model()/MockInferenceAdapter.deploy_llm_model()
        so factory.py can hand any of the three to routers/llm_serving.py
        without it noticing. Not implemented: GPU/vLLM serving is
        explicitly out of this phase's real-verification scope
        (docs/openchoreo-migration-next-steps.md Phase 4 — "GPU/vLLM
        (deploy_llm_model) không nằm trong phạm vi verify thật của Phase
        này"), and there's no ClusterComponentType for it yet (the one
        real ClusterComponentType this session built, "proxy/inference-service",
        is MLflow-CPU-only — see its own file's header)."""
        del (
            name,
            version,
            huggingface_model_id,
            serving_runtime_name,
            gpu_count,
            vllm_quantization,
            max_context_length,
            traffic_fields,
        )
        raise NotImplementedError(
            "OpenChoreoInferenceAdapter.deploy_llm_model: no ClusterComponentType "
            "for GPU/vLLM serving exists yet — out of Phase 4's scope"
        )

    def get_inference_status(self, name: str) -> dict[str, object]:
        del name  # same single-Workload scoping as deploy_model
        # The real InferenceService lives in an auto-generated dataplane
        # namespace (e.g. dp-default-fraud-detecti-development-<hash>) that
        # can't be derived from `self.project`/`self.component` — only
        # discovered by listing across all namespaces, filtered by the
        # labels OpenChoreo's renderedrelease-controller actually sets
        # (confirmed via `kubectl get inferenceservice ... -o
        # jsonpath='{.metadata.labels}'`).
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

    def predict(self, name: str, payload: dict[str, object]) -> dict[str, object]:
        del name, payload
        raise NotImplementedError(
            "Call the InferenceService's HTTP endpoint directly "
            "(get the URL from get_inference_status) instead of going through this adapter"
        )
