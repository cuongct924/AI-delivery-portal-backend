# llm-serving

Self-hosted LLM serving (vLLM on KServe) powering the "Serving LLM" Golden
Path (`templates/llm-serve-deploy/template.yaml` in the frontend repo,
`cuongct924/AI-delivery-portal-frontend`). **Does not work
locally** — the k3d `openchoreo-quick-start` cluster used by every other
Golden Path in this repo has no GPU node, no NVIDIA device plugin, and no
`huggingface`/vLLM `ServingRuntime` — only the `mlflow` runtime KServe
ships by default. There is also no OpenChoreo `ClusterComponentType` for
GPU/vLLM serving — `adapters/delivery/gpu_inference_adapter.py`'s `GpuKServeInferenceAdapter`
talks to KServe's InferenceService CRD directly, deliberately kept scoped to
this one golden path while standard model serving (`routers/models.py`) is
fully on OpenChoreo (`adapters/delivery/openchoreo_inference_adapter.py`).

`adapters/delivery/gpu_inference_adapter.py`'s `deploy_llm_model()` and
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
   `ClusterComponentType` for GPU/vLLM serving (doesn't exist yet).
3. **Roadmap runtimes** — the dropdown also lists `tensorrt-llm` and `triton`
   (mapped to `tensorrt-llm-runtime` / `triton-runtime` ClusterServingRuntimes
   respectively) but these raise friendly errors at prepare time in the current
   MVP. See the Runtime/Optimization Support Matrix below.

## Runtime/Optimization Support Matrix — Implemented vs Roadmap

| Runtime | ServingRuntime CR | Continuous Batching | PagedAttention | Prefix Caching | Speculative Decoding | Disaggregation | KV Offload | Pipeline Parallel |
|---------|-------------------|---------------------|----------------|----------------|----------------------|----------------|------------|-------------------|
| vLLM | `vllm-runtime` | ✅ Default | ✅ Default | ✅ Configurable | 🔄 Roadmap (MVP: blocked) | 🔄 Roadmap | 🔄 Roadmap | ✅ Configurable |
| tensorrt-llm | `tensorrt-llm-runtime` | 🔄 Roadmap | 🔄 Roadmap | 🔄 Roadmap | 🔄 Roadmap | 🔄 Roadmap | 🔄 Roadmap | 🔄 Roadmap |
| triton | `triton-runtime` | 🔄 Roadmap | 🔄 Roadmap | 🔄 Roadmap | 🔄 Roadmap | 🔄 Roadmap | 🔄 Roadmap | 🔄 Roadmap |

- ✅ = Implemented and validated in MVP (vLLM on cluster demo)
- 🔄 = Roadmap — UI dropdown shows the option but prepare-manifest fails with a friendly error pointing here

**Default-on optimizations** (vLLM): continuous batching + PagedAttention are enabled by default in vLLM — no action needed by the Dev. The form's `enablePagedAttention=true` / `batchingStrategy=continuous` reflect this.

**Must configure + benchmark** (vLLM): prefix caching (`enablePrefixCaching`), pipeline parallelism (`pipelineParallelSize>1`). These are exposed in the form but the Dev must test them for their specific model/workload.

**Roadmap optimizations** (all runtimes): speculative decoding (ngram / draft-model), prefill-decode disaggregation, KV cache offload (CPU/SSD), expert/data parallelism beyond tensor-parallel. These require separate infrastructure (draft model serving, prefill/decode pools, offload storage) and are intentionally excluded from MVP.

## Why quantization/GPU compatibility is enforced in code, not just this doc

`llm_serving/registry.py`'s `validate_gpu_quantization()` is the real
guard (raises before rendering anything) — the Scaffolder form's JSON
Schema `allOf`/`if`/`then` narrowing is UX only, not enforcement (CLAUDE.md:
business logic lives in orchestration-api).

## Optimization validation

`llm_serving/registry.py`'s `validate_runtime_optimizations()` rejects any
optimization flag not in the runtime's supported set with a clear message:
```
Optimization 'speculativeDecoding' is not supported for runtime 'vllm'
in this MVP — supported flags for 'vllm': ['batchingStrategy', 'enablePagedAttention', 'enablePrefixCaching', 'pipelineParallelSize'].
See infra/llm-serving/README.md for the roadmap.
```

This keeps the form honest: it shows all flags, but the backend tells you
exactly which ones work today.
