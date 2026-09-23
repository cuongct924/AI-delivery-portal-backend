# Troubleshooting FAQ

Real, known issues and how to reason about them — not a generic checklist.
Each entry says what you'll observe, why it happens, and what to actually
do about it.

## A deployed model's InferenceService never becomes Ready

**Symptom:** you ran Evaluate & Deploy, the deploy step succeeded, but the
InferenceService stays stuck (never loads a model, `kubectl get
inferenceservice <name> -o yaml` shows it waiting).

**Why:** the platform's model URIs use the `models:/<name>/<version>`
convention (MLflow Model Registry shorthand). KServe's storage-initializer
only recognizes `gs://`, `s3://`, `file://`, `http(s)://`, and `hf://` — it
cannot resolve `models:/` on its own, and the in-cluster MLflow service
fronts its artifacts behind its own `mlflow-artifacts:/` proxy, which
doesn't help either. This is a known, documented, currently-unresolved gap
(not something you misconfigured) — see
`adapters/delivery/openchoreo_inference_adapter.py`'s module docstring.

**What to do:** there is no clean workaround yet. The real fix needs either
switching the MLflow artifact store to something KServe's initializer can
read directly (e.g. S3/MinIO with a `s3://` URI), or resolving
`model_uri` to a concrete URI before it reaches the manifest. Flag it rather
than assuming your deploy step did something wrong.

## Setting up periodic drift monitoring for a prompt/RAG index doesn't do anything

**Symptom:** you'd expect "Set Up Model Monitoring"-style periodic checks to
also exist for the LLMOps track (prompts/RAG indexes), but nothing runs on
a schedule.

**Why:** `OpenChoreoWorkflowAdapter.create_cron_workflow()` explicitly
raises `NotImplementedError` — OpenChoreo's Workflow/WorkflowRun API has no
CronWorkflow equivalent yet. The intended replacement (a scheduled-task
Component whose per-environment schedule comes from a ReleaseBinding) is
blocked on a `ReleaseBinding`-creation call this adapter doesn't make yet.

**What to do:** this is a platform capability gap, not a bug in your
config. Periodic MLOps drift monitoring (the plain model-metrics case)
still works — it's specifically the *cron/schedule* mechanism for LLMOps
that isn't wired up.

## Self-hosted LLM serving (vLLM) never shows up as an OpenChoreo Component

**Symptom:** models deployed via the standard Evaluate & Deploy path show
up as OpenChoreo Components/Workloads, but a self-hosted LLM deployed via
"Serving LLM" doesn't.

**Why:** this is by design, not a bug. No OpenChoreo ClusterComponentType
exists yet for GPU/vLLM workloads, so `IGpuInferenceAdapter` has no
`"openchoreo"` backend mode — it only ever talks to KServe directly
(`GpuKServeInferenceAdapter`), the same way standard model serving used to
before it fully migrated.

## A `USE_MOCK_WORKFLOW=legacy` or `USE_MOCK_INFERENCE=legacy` setting raises an error at startup

**Why:** both the training/workflow path and the standard model-serving
path have fully migrated to OpenChoreo — the old direct-to-Argo and
direct-to-KServe adapters were removed once that migration finished.
`"legacy"` is a deliberately-rejected value now, not a silent no-op.

**What to do:** set the env var to `"openchoreo"` (the real backend) or
`"mock"` (for local/demo use without a live cluster).

## A drafted prompt/RAG version passed evaluation locally but the live chat assistant still gives old answers

**Why:** activation is tracked **independently per environment**
(`development` / `staging` / `production`) since the LLMOps environment
feature was added. `routers/chat.py` always reads from `"production"`
specifically. If you activated in `"development"` or `"staging"` to try it
first, that's expected — the change hasn't reached production yet.

**What to do:** re-run "Evaluate & Activate Prompt / RAG Version" with
`targetEnvironment=production` (or use `action=rollback` if you need to
revert something already in production, without waiting on the gate again).

## A training run gets blocked by "likely leakage" or a missing-value error before it even starts

**Why:** `data_quality/checks.py` runs before training and blocks (not just
warns) on two conditions: the target column has missing values, or a
feature is correlated ≥ 0.98 with the target (`check_target_leakage_correlation`
— almost certainly the feature encodes the label itself). A correlation
between 0.9 and 0.98 is only a warning, not a block.

**What to do:** this usually means a real data problem, not a check to
bypass — a near-perfect correlation with the target is the textbook
signature of a column that shouldn't be a feature at all (e.g. it was
computed *from* the label).
