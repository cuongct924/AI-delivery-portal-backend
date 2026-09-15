# RQ1 — Manual baseline (Lead Time comparison)

`infra/monitoring/grafana/dashboards/dora-metrics.json`'s "Lead Time"
panel measures the IDP side of RQ1 automatically, from real Argo Workflow
timestamps (see `observability/dora_metrics.py`). The "Manual" side has no
automated equivalent — it has to be timed by hand, once, following the
steps below, and the result recorded here.

## Method

Time a single Golden Path #1 (Train → Track → Register) run performed
entirely by hand, with a stopwatch started the moment the dataset/code is
ready and stopped the moment the model version is visible in the MLflow UI:

1. Write a Kubernetes `Job` manifest (or Argo `Workflow` manifest) by hand
   for the training container — no `WorkflowTemplate`, no Scaffolder Form.
2. Write/adjust the training image's `Dockerfile` if anything about the
   dataset/architecture isn't already covered by the existing image.
3. Set the env vars / dataset URI / hyperparameters by hand in the
   manifest (the same fields the Scaffolder Form fills in for you).
4. `kubectl apply` the manifest, `kubectl logs -f` to watch it run.
5. On completion, manually register the model version in MLflow (`mlflow
   models register` or the UI) if the script didn't already call
   `mlflow.log_model`/`register_model` itself.
6. Stop the stopwatch once the new version is visible in MLflow's UI.

## Result

| Run | Date | Total time (manual) | Notes |
|---|---|---|---|
| _(not yet run)_ | | | |

Fill this table in after actually performing the manual run above — do not
estimate or infer a number, this repo's own convention (see
`infra/openchoreo/README.md`) is to only report measurements that were
actually taken.

## Caveat

Golden Path #2 (Register → Deploy)'s Lead Time, as currently measured by
the dashboard, reflects `MockInferenceAdapter`'s synthetic timing —
`USE_MOCK_INFERENCE=true` today (see `.env`). Do not present that number as
real infrastructure Lead Time until Phase 3/4 of the OpenChoreo plan
(`~/.claude/plans/goofy-growing-dijkstra.md`, Phần B) lands a real KServe
adapter.
