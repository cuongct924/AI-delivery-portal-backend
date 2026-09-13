"""Mock adapter for IInferenceAdapter — in-memory stand-in for KServe.

Raises `ApiException(status=404)` from `get_inference_status()` for an
unknown model, same as the real KServe client, so callers need no changes.
"""

from collections.abc import Mapping

from kubernetes.client.exceptions import ApiException

from adapters.interfaces import IInferenceAdapter


class MockInferenceAdapter(IInferenceAdapter):
    def __init__(self, namespace: str = "default") -> None:
        self.namespace = namespace
        self._deployed: dict[str, dict[str, object]] = {}

    def deploy_model(
        self,
        name: str,
        version: str,
        model_uri: str,
        traffic_fields: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        status = self._render_status(name, version, model_uri, traffic_fields)
        self._deployed[name] = status
        return status

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
        KServeAdapter.deploy_llm_model() so factory.py can hand either one
        to routers/llm_serving.py without it noticing."""
        del serving_runtime_name, gpu_count, vllm_quantization, max_context_length
        status = self._render_status(name, version, f"hf://{huggingface_model_id}", traffic_fields)
        self._deployed[name] = status
        return status

    def get_inference_status(self, name: str) -> dict[str, object]:
        status = self._deployed.get(name)
        if status is None:
            raise ApiException(status=404, reason=f"InferenceService {name} not found")
        return status

    def predict(self, name: str, payload: dict[str, object]) -> dict[str, object]:
        if name not in self._deployed:
            raise ApiException(status=404, reason=f"InferenceService {name} not found")
        return {"predictions": [0], "echo": payload}

    def _render_status(
        self,
        name: str,
        version: str,
        storage_uri: str,
        traffic_fields: Mapping[str, object] | None,
    ) -> dict[str, object]:
        return {
            "metadata": {"name": name, "labels": {"version": version}},
            "spec": {"predictor": {**(traffic_fields or {}), "model": {"storageUri": storage_uri}}},
            "status": {
                "url": f"http://{name}.{self.namespace}.mock.svc.cluster.local",
                "conditions": [{"type": "Ready", "status": "True"}],
            },
        }
