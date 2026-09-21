"""Adapter for KServe, scoped exclusively to the LLM self-hosted-serving
golden path (routers/llm_serving.py). Not used by, and must never be
reintroduced into, the standard model-serving golden path
(routers/models.py), which is fully on OpenChoreoInferenceAdapter — there is
no OpenChoreo ClusterComponentType for GPU/vLLM serving yet, so this is the
one place a real backend still talks to KServe's `InferenceService` custom
resource (serving.kserve.io/v1beta1) directly via the Kubernetes API.
Requires a kubeconfig pointing at a real cluster.
"""

from collections.abc import Mapping
from typing import Final, cast

from kubernetes import client
from kubernetes.client.exceptions import ApiException

from adapters.delivery._kube_client import load_kube_config_once
from adapters.delivery.interfaces import DeployStatus, IGpuInferenceAdapter

GROUP: Final[str] = "serving.kserve.io"
VERSION: Final[str] = "v1beta1"
PLURAL: Final[str] = "inferenceservices"


class GpuKServeInferenceAdapter(IGpuInferenceAdapter):
    def __init__(self, namespace: str = "default"):
        load_kube_config_once()
        self.namespace = namespace
        self.api = client.CustomObjectsApi()

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
        hf_token_secret_ref: str | None = None,
    ) -> dict[str, object]:
        """Patch-first/create-on-404 for a self-hosted LLM — modelFormat
        "huggingface" (KServe's own vLLM-backed runtime), storageUri as an
        "hf://" reference.

        vllm_quantization is vLLM's own --quantization value (translated from
        the Dev-facing label by llm_serving.registry.VLLM_QUANTIZATION_ARGS —
        adapters/ can't import from services/orchestration-api/). hf_token_secret_ref
        is a K8s Secret name (key "token"), never a plaintext value."""
        args = [
            f"--tensor-parallel-size={gpu_count}",
            f"--max-model-len={max_context_length}",
        ]
        if vllm_quantization is not None:
            args.append(f"--quantization={vllm_quantization}")
        model_spec: dict[str, object] = {
            "modelFormat": {"name": "huggingface"},
            "runtime": serving_runtime_name,
            "storageUri": f"hf://{huggingface_model_id}",
            "args": args,
            "resources": {
                "requests": {"nvidia.com/gpu": str(gpu_count)},
                "limits": {"nvidia.com/gpu": str(gpu_count)},
            },
        }
        if hf_token_secret_ref is not None:
            model_spec["env"] = [
                {
                    "name": "HUGGING_FACE_HUB_TOKEN",
                    "valueFrom": {"secretKeyRef": {"name": hf_token_secret_ref, "key": "token"}},
                }
            ]
        body = {
            "apiVersion": f"{GROUP}/{VERSION}",
            "kind": "InferenceService",
            "metadata": {"name": name, "labels": {"version": version}},
            "spec": {"predictor": {**(traffic_fields or {}), "model": model_spec}},
        }
        try:
            result = self.api.patch_namespaced_custom_object(
                GROUP, VERSION, self.namespace, PLURAL, name, body
            )
        except ApiException as exc:
            if exc.status != 404:
                raise
            result = self.api.create_namespaced_custom_object(
                GROUP, VERSION, self.namespace, PLURAL, body
            )
        # cast: only non-dict when async_req=True, which we never pass.
        return cast(dict[str, object], result)

    def get_inference_status(self, name: str) -> dict[str, object]:
        return cast(
            dict[str, object],
            self.api.get_namespaced_custom_object_status(
                GROUP, VERSION, self.namespace, PLURAL, name
            ),
        )

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

        metadata = cast(dict[str, object], status.get("metadata", {}))
        labels = cast(dict[str, object], metadata.get("labels") or {})
        # deploy_llm_model sets this label — the only version info that survives.
        live_version = cast(str | None, labels.get("version"))

        conditions = cast(
            list[dict[str, object]],
            cast(dict[str, object], status.get("status", {})).get("conditions", []),
        )
        ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)

        return DeployStatus(
            deployed=True, ready=ready, live_version=live_version, traffic_percent=traffic_percent
        )

    def predict(self, name: str, payload: dict[str, object]) -> dict[str, object]:
        raise NotImplementedError(
            "Call the InferenceService's HTTP endpoint directly "
            "(get the URL from get_inference_status) instead of going through this adapter"
        )
