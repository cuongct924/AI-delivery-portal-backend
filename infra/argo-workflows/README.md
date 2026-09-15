# argo-workflows

1 `WorkflowTemplate` for Golden Path #1 (Train → Track → Register),
triggered by `adapters/argo_adapter.py` via the Argo Server REST API:

- `train-register-template.yaml` (`train-register-golden-path`) — a
  `mode` parameter (`train` default, `finetune`) picks between training
  from scratch and fine-tuning `base-model-uri`, both on the same DAG.

`train-step` runs `training-image/` (built from `Dockerfile` in that
directory — dependencies installed at image build time, not pip-installed
per job run, which used to hit a real Docker Desktop I/O error mid-download).
It reads `TASK_TYPE`/`ALGORITHM`/`TARGET_COLUMN`/`ID_COLUMNS`/`TIME_COLUMN`
env vars, trains via `algorithm_registry.py` (~18 sklearn/XGBoost/LightGBM/
CatBoost algorithms), logs metrics + dataset lineage + the model to MLflow,
then hands the resulting `runs:/...` artifact URI and dataset digest to
`register-step`, which calls `POST /models/register` on orchestration-api
(business logic — the actual MLflow registration — stays there, not in the
workflow pod, per CLAUDE.md). This runs on Argo Workflows inside the k3d
`openchoreo-quick-start` cluster (its `openchoreo-workflow-plane` build
plane) — see the top-level `README.md`'s "Local Kubernetes cluster" section
for the one-time RBAC/WorkflowTemplate/image-import setup. How the cluster
itself gets created isn't documented in this repo yet — see
`docs/openchoreo-migration-next-steps.md` Phase 0 for context.

**`command: ["python", "train.py"]` on the train-step is required, not
cosmetic**: `training-image:local` is only ever built + `k3d image import`ed,
never pushed to a real registry. Without an explicit `command`, Argo's
emissary executor tries to look up the image's default entrypoint by
querying its registry (falls back to `docker.io/library/...` for an
unqualified name) and fails with an `UNAUTHORIZED` manifest-pull error even
though the image already exists on the node. Re-run the build +
`k3d image import` step (top-level `README.md`) after any change under
`training-image/`.

**Reaching host-machine services from a workflow step container:** use
`http://host.docker.internal:5000` (mlflow) and
`http://host.docker.internal:8000` (orchestration-api), not `localhost` —
step containers run as pods on the k3d server node, which is itself a
Docker container. This resolves correctly under Docker Desktop for Mac (the
repo owner's OS is Darwin/macOS); other Docker setups (e.g. Docker Desktop
for Linux, `colima`) may need a different host address.
