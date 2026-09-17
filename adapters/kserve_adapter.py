"""Adapter for KServe — deploys/queries models as InferenceService on K8s.

Uses the Kubernetes Python client to operate on the `InferenceService`
custom resource (serving.kserve.io/v1beta1). Requires a kubeconfig pointing
at a real cluster.
"""

from collections.abc import Mapping
from typing import Final, cast

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from adapters.interfaces import IInferenceAdapter

GROUP: Final[str] = "serving.kserve.io"
VERSION: Final[str] = "v1beta1"
PLURAL: Final[str] = "inferenceservices"


class KServeAdapter(IInferenceAdapter):
    def __init__(self, namespace: str = "default"):
        config.load_kube_config()
        self.namespace = namespace
        self.api = client.CustomObjectsApi()

    def deploy_model(
        self,
        name: str,
        version: str,
        model_uri: str,
        traffic_fields: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Creates the InferenceService, or patches it if a version was
        already deployed under this name (patch first — the common case
        for adapters/deploy_strategies.py's TrafficSplitStrategy, which by
        definition requires a prior deploy to exist)."""
        body = {
            "apiVersion": f"{GROUP}/{VERSION}",
            "kind": "InferenceService",
            "metadata": {"name": name, "labels": {"version": version}},
            "spec": {
                "predictor": {
                    **(traffic_fields or {}),
                    "model": {
                        "modelFormat": {"name": "mlflow"},
                        "storageUri": model_uri,
                    },
                }
            },
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
        """Same patch-first/create-on-404 shape as deploy_model(), for a
        self-hosted LLM — modelFormat "huggingface" (KServe's own vLLM-backed
        runtime), storageUri as an "hf://" reference. Not part of
        IInferenceAdapter — its deploy_model() has no room for
        GPU/quantization/context-length; same precedent as get_latest_version().

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

    def predict(self, name: str, payload: dict[str, object]) -> dict[str, object]:
        raise NotImplementedError(
            "Call the InferenceService's HTTP endpoint directly "
            "(get the URL from get_inference_status) instead of going through this adapter"
        )
