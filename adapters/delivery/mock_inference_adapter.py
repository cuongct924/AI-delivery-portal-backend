"""Mock adapter for both IInferenceAdapter (standard serving) and
IGpuInferenceAdapter (LLM self-hosted serving) — in-memory stand-in for
OpenChoreo/KServe. One class implements both: both real backends need an
equivalent mock, and they already share the exact same in-memory shape/
_render_status helper, so a shared mock avoids duplicating it.

Raises `ApiException(status=404)` from `get_inference_status()` for an
unknown model, same as the real KServe client, so callers need no changes.
"""

from collections.abc import Mapping

from kubernetes.client.exceptions import ApiException

from adapters.delivery.interfaces import DeployStatus, IGpuInferenceAdapter, IInferenceAdapter


class MockInferenceAdapter(IInferenceAdapter, IGpuInferenceAdapter):
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
        hf_token_secret_ref: str | None = None,
    ) -> dict[str, object]:
        """Mirrors GpuKServeInferenceAdapter.deploy_llm_model() so factory.py
        can hand either one to routers/llm_serving.py without it noticing."""
        del serving_runtime_name, gpu_count, vllm_quantization, max_context_length
        del hf_token_secret_ref
        status = self._render_status(name, version, f"hf://{huggingface_model_id}", traffic_fields)
        self._deployed[name] = status
        return status

    def get_inference_status(self, name: str) -> dict[str, object]:
        status = self._deployed.get(name)
        if status is None:
            raise ApiException(status=404, reason=f"InferenceService {name} not found")
        return status

    def get_deploy_status(self, name: str) -> DeployStatus:
        try:
            status = self.get_inference_status(name)
        except ApiException as exc:
            if exc.status == 404:
                return DeployStatus(
                    deployed=False, ready=False, live_version=None, traffic_percent=None
                )
            raise

        spec = status["spec"] if isinstance(status["spec"], dict) else {}
        predictor = spec["predictor"] if isinstance(spec.get("predictor"), dict) else {}
        traffic_percent = predictor.get("canaryTrafficPercent")

        metadata = status["metadata"] if isinstance(status["metadata"], dict) else {}
        labels = metadata["labels"] if isinstance(metadata.get("labels"), dict) else {}
        live_version = labels.get("version")

        status_block = status["status"] if isinstance(status["status"], dict) else {}
        conditions = (
            status_block["conditions"] if isinstance(status_block.get("conditions"), list) else []
        )
        ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)

        return DeployStatus(
            deployed=True, ready=ready, live_version=live_version, traffic_percent=traffic_percent
        )

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
