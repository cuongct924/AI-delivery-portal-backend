# llm-serving

Self-hosted LLM serving (vLLM on KServe) powering the "Serving LLM" Golden
Path (`examples/templates/deploy-llm/template.yaml`). **Does not work
locally** — the `kind` cluster used by every other Golden Path in this repo
(`infra/k8s-local-cluster/kind-config.yaml`) has no GPU node, no NVIDIA
device plugin, and no `huggingface`/vLLM `ServingRuntime` — only the
`mlflow` runtime KServe ships by default. KServe itself also isn't
installed anywhere yet (see `infra/openchoreo/README.md` — no
ClusterComponentType wraps it), a separate prerequisite from the GPU gap
below.

`adapters/kserve_adapter.py`'s `deploy_llm_model()` and
`services/orchestration-api/routers/llm_serving.py`'s
`/llm-deploy/prepare` render a correct `InferenceService` manifest and are
covered by real (non-GPU) unit tests — what's genuinely not implemented
here is the cluster-side prerequisite below.

## What a real GPU cluster needs before this Golden Path can actually deploy

1. **A GPU node pool** — at least 1 node with an NVIDIA GPU (see
   `services/orchestration-api/llm_serving/registry.py`'s
   `GPU_QUANTIZATION_COMPATIBILITY` for the supported types: L4, L40S,
   A100, H100, H200, B200) and the
   [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin)
   DaemonSet installed, so `nvidia.com/gpu` resource requests actually
   schedule.
2. **A `vllm-runtime` ClusterServingRuntime** registered in KServe —
   reference shape (confirmed against
   [kserve.github.io/website/docs/getting-started/genai-first-isvc](https://kserve.github.io/website/docs/getting-started/genai-first-isvc)):
   ```yaml
   apiVersion: serving.kserve.io/v1alpha1
   kind: ClusterServingRuntime
   metadata:
     name: vllm-runtime
   spec:
     supportedModelFormats:
       - name: huggingface
         autoSelect: true
     containers:
       - name: kserve-container
         image: kserve/huggingfaceserver:latest # pin a real tag before use
   ```
   Not applied by any script in this repo — apply it by hand on a real
   GPU cluster once KServe itself is wrapped in an OpenChoreo
   `ClusterComponentType` (Phase 2, sub-step 2.3).

## Why quantization/GPU compatibility is enforced in code, not just this doc

`llm_serving/registry.py`'s `validate_gpu_quantization()` is the real
guard (raises before rendering anything) — the Scaffolder form's JSON
Schema `allOf`/`if`/`then` narrowing is UX only, not enforcement (CLAUDE.md:
business logic lives in orchestration-api).
