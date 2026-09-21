# AI Delivery Portal

**An Internal Developer Platform for MLOps/LLMOps**

Repository for the **AI Delivery Portal** project (Viettel Digital Talent 2026 — Cloud Track - Phase 02).

An MLOps/LLMOps platform built as an Internal Developer Platform (IDP) for AI/ML workloads:

- Acts as the **DevEx and orchestration layer** on top of the AI platform ecosystem — Model Registry, Model Experiments, AI Inference, AI Notebooks
- Provides **golden paths** so developers and ML engineers can ship standard MLOps/LLMOps workflows without hand-rolling infrastructure each time
- Scope is the internal MLOps/LLMOps flow itself — integrating generic CI/CD with external systems is out of scope (see [`docs/playbook-ai-delivery-portal.md`](docs/playbook-ai-delivery-portal.md))

Official structure: Portal (Backstage) + Orchestration API (FastAPI) + AI Agent/MCP
+ Adapter layer + GitOps infrastructure, split across two repos: this one is
the **backend**; the Portal frontend (Backstage + OpenChoreo plugins +
Viettel Cloud branding) lives in a separate repo,
[`cuongct924/AI-delivery-portal-frontend`](https://github.com/cuongct924/AI-delivery-portal-frontend).

## Directory structure

```
AI-delivery-portal/
├── services/             ← orchestration-api — FastAPI BFF, MCP client, auth, evaluations
├── agents/               ← AI Agent & MCP — mcp-servers/, skills/, prompts/
├── adapters/             ← Adapter Pattern — MLflow, KServe, Argo, Qdrant, LiteLLM, Feast, JupyterHub
├── infra/                ← GitOps infra — monitoring/vector-dbs/llm-gateways (active); helm-charts/argocd/opa-policies (not yet implemented)
├── data/                 ← datasets versioned with DVC (S3-compatible remote, see data/README.md)
├── scripts/              ← run-mcp-local.sh
├── docs/                 ← playbook, LLMOps draft plan
├── docker-compose.yml    ← full local stack (mlflow, prometheus, grafana, qdrant, minio, litellm, orchestration-api, MCP servers)
└── README.md
```

Portal frontend (Golden Path Scaffolder templates, Catalog entities, custom
Scaffolder actions/fields) lives entirely in the separate frontend repo —
not here.

See [`docs/playbook-ai-delivery-portal.md`](docs/playbook-ai-delivery-portal.md) for the full component breakdown and design rationale.

## Usage

```bash
cp .env.example .env            # fill in ANTHROPIC_API_KEY before running orchestration-api/litellm
docker compose --profile mlops up -d       # MLOps only: mlflow, minio, orchestration-api
docker compose --profile llmops up -d      # LLMOps only: qdrant, litellm
docker compose --profile observability up -d   # optional: prometheus, grafana
docker compose --profile mlops --profile llmops --profile observability up -d   # everything

# MCP servers run detached, discoverable via the Backstage Catalog:
docker compose up -d observability-server llmops-golden-paths-server mlops-golden-paths-server
bash scripts/run-mcp-local.sh observability   # or: run a single MCP server standalone, no Docker needed

make install    # create a Python 3.12 .venv, install ruff + pyright + pytest + every service's requirements.txt
make lint       # ruff check .
make format     # ruff format .
make typecheck  # pyright
make test       # pytest (tests/)
make check      # lint + typecheck + test
```

- Portal UI + Golden Path templates/Catalog config: see the frontend repo,
  [`cuongct924/AI-delivery-portal-frontend`](https://github.com/cuongct924/AI-delivery-portal-frontend)
  — [`orchestration-api`](services/orchestration-api/) must be running
  (`docker compose up` or local `uvicorn`) for its Scaffolder actions to work

### Orchestration API (FastAPI)

```bash
make install                  # one-time: creates .venv + installs every service's deps
make run-orchestration-api    # uvicorn main:app --reload on http://localhost:8000
curl http://localhost:8000/healthz   # {"status": "ok"}
```

Or skip the local Python env entirely and use `docker compose up -d`, which runs
the same service (plus its MCP neighbours) from `docker-compose.yml`.

### Adapters

Every external integration (MLflow, KServe, OpenChoreo, Qdrant, LiteLLM,
Feast, JupyterHub, ...) sits behind an interface in
[`adapters/delivery/interfaces.py`](adapters/delivery/interfaces.py) (deploy/
promote/workflow) or [`adapters/ai_platform/interfaces.py`](adapters/ai_platform/interfaces.py)
(registry/experiments/feature-store/vector-store/gateway);
[`adapters/factory.py`](adapters/factory.py) picks the concrete class from the
`USE_MOCK_*` env vars (see `.env.example`), so switching Mock → real backend
means adding a class, never touching callers.
Business logic stays in `services/orchestration-api/`, not in workflow pods or
the Portal frontend.

### LiteLLM gateway

```bash
docker compose --profile llmops up -d litellm   # http://localhost:4000
```

Models and routing are defined in
[`infra/ai-platform-zone/litellm-config.yaml`](infra/ai-platform-zone/litellm-config.yaml)
(e.g. `claude-sonnet-5`, `voyage-3`, `llama3.1-local`); set
`ANTHROPIC_API_KEY`/`VOYAGE_API_KEY` in `.env` as needed.

### Test CI locally

[`act`](https://github.com/nektos/act) runs [`.github/workflows/ci.yml`](.github/workflows/ci.yml) itself, on your machine, before you push:

```bash
# For MacOS only
brew install act
echo "--container-architecture linux/amd64" >> ~/.actrc   # Apple Silicon: match the amd64 GitHub-hosted runner

act -l                                                      # list jobs, sanity-check the workflow parses
act pull_request --container-architecture linux/amd64       # run the full pipeline (skips the GHCR push step — needs main branch)
act pull_request -j python-checks                           # run a single job: python-checks example
```

## Local Kubernetes cluster

Golden Path #1 runs on Argo Workflows inside the same k3d `openchoreo-quick-start`
cluster used for OpenChoreo/Thunder — there is no separate local cluster for
it anymore (cluster creation itself isn't documented in this repo yet).
Setup (one-time, after the cluster exists):

```bash
kubectl config use-context k3d-openchoreo-quick-start

# ServiceAccount + RBAC for the retained legacy-style templates
# (monitor-drift-template.yaml). The
# OpenChoreo production path does NOT need this — it uses the auto-provisioned
# workflow-sa in its per-namespace execution namespace instead.
kubectl -n default apply -f - <<'EOF'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: train-register-workflow
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: train-register-workflow
rules:
  - apiGroups: ["argoproj.io"]
    resources: ["workflowtaskresults"]
    verbs: ["create", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: train-register-workflow
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: train-register-workflow
subjects:
  - kind: ServiceAccount
    name: train-register-workflow
    namespace: default
EOF

# OpenChoreo workflow resources: ClusterWorkflow + its cluster-scoped
# execution template, for Golden Path #1
kubectl apply -f infra/openchoreo/platform-shared/cluster-workflow-templates/argo/train-register-cluster-template.yaml
kubectl apply -f infra/openchoreo/platform-shared/workflows/clusterworkflow-training.yaml

# training-image isn't pushed to a registry — build it, then import into
# the cluster's containerd. Re-run after any change under
# infra/argo-workflows/training-image/.
docker build -t training-image:local -f infra/argo-workflows/training-image/Dockerfile .
k3d image import training-image:local --cluster openchoreo-quick-start

# the dataset hostPath (train-register-cluster-template.yaml) needs data/ copied
# into the k3d server node's filesystem — no bind-mount for an
# already-running node, so this is a one-time (or after dvc pull) docker cp:
docker cp data/. k3d-openchoreo-quick-start-server-0:/mnt/data/
```

Two non-obvious execution requirements apply to the workflow step pods rendered
by the `ClusterWorkflowTemplate`s above:

- `train-step` must set `command: ["python", "train.py"]` explicitly. The
  `training-image:local` image is only built + `k3d image import`ed, never pushed
  to a registry, so without an explicit `command` Argo's emissary executor
  queries the registry for the image entrypoint and fails with an `UNAUTHORIZED`
  manifest-pull error even though the image exists on the node.
- From a step container, reach host-machine services via
  `http://host.docker.internal:5000` (mlflow) and
  `http://host.docker.internal:8000` (orchestration-api), not `localhost` — step
  pods run on the k3d server node, which is itself a Docker container.

Day-to-day:

```bash
# cluster is up if this returns a Ready node
kubectl get nodes
# Argo controller/server status (namespace is OpenChoreo's build/workflow plane)
kubectl get pods -n openchoreo-workflow-plane
# controller/server logs
kubectl logs -n openchoreo-workflow-plane -l app=workflow-controller
kubectl logs -n openchoreo-workflow-plane -l app=argo-server

kubectl get clusterworkflows,clusterworkflowtemplates   # confirm the train/rec ClusterWorkflow + templates exist
kubectl get workflowruns -n default                     # list runs triggered via POST /trigger-training
kubectl get workflows -n workflows-default -w           # watch a run's rendered phase live
kubectl logs -n workflows-default <pod-name>            # logs of a specific train/register step pod

curl http://localhost:10081/api/v1/workflows/default    # Argo Server REST API health check (the workflow engine OpenChoreo WorkflowRuns run on)

k3d cluster delete openchoreo-quick-start               # tear the whole cluster down (also takes out Thunder/OpenChoreo)
```

## Component guides

- [data/README.md](data/README.md) — DVC workflow, dataset layout, dataset contracts.
- [infra/feature-store/README.md](infra/feature-store/README.md) — Feast init + online/offline store setup.
- [infra/llm-serving/README.md](infra/llm-serving/README.md) — GPU/KServe/vLLM prerequisites + capability matrix.
- [infra/monitoring/README.md](infra/monitoring/README.md) — Prometheus/Grafana setup + monitoring limitations.
- [agents/mcp-servers/golden-path-guide-server/README.md](agents/mcp-servers/golden-path-guide-server/README.md) — Golden Path discovery MCP server.
- [agents/mcp-servers/llmops-golden-paths-server/README.md](agents/mcp-servers/llmops-golden-paths-server/README.md) — LLMOps Golden Paths MCP server.
- [agents/mcp-servers/mlops-golden-paths-server/README.md](agents/mcp-servers/mlops-golden-paths-server/README.md) — MLOps Golden Paths MCP server.
- [agents/mcp-servers/observability-server/README.md](agents/mcp-servers/observability-server/README.md) — Prometheus/MLflow/promotion-status MCP server.

## Reference

- Design decisions (golden path, tech stack, benchmarks, mentor questions):
  [`docs/playbook-ai-delivery-portal.md`](docs/playbook-ai-delivery-portal.md).
- Golden Path #1/#3 workflow backend migration:
  [`docs/openchoreo-workflow-migration-plan.md`](docs/openchoreo-workflow-migration-plan.md).
- OpenChoreo environments and promotion policy:
  [`docs/openchoreo-environments.md`](docs/openchoreo-environments.md).
