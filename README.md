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
docker compose up -d observability-server golden-paths-server
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
cluster used for OpenChoreo/Thunder (see `docs/openchoreo-migration-next-steps.md`
for its current state — cluster creation itself isn't documented in this repo
yet) — there is no separate local cluster for it anymore. Setup (one-time,
after the cluster exists):

```bash
kubectl config use-context k3d-openchoreo-quick-start

# ServiceAccount + RBAC for the workflow pods (see infra/argo-workflows/train-register-template.yaml)
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

kubectl apply -f infra/argo-workflows/train-register-template.yaml

# training-image isn't pushed to a registry — build it, then import into
# the cluster's containerd. Re-run after any change under
# infra/argo-workflows/training-image/.
docker build -t training-image:local -f infra/argo-workflows/training-image/Dockerfile .
k3d image import training-image:local --cluster openchoreo-quick-start

# the dataset hostPath (train-register-template.yaml) needs data/ copied
# into the k3d server node's filesystem — no bind-mount for an
# already-running node, so this is a one-time (or after dvc pull) docker cp:
docker cp data/. k3d-openchoreo-quick-start-server-0:/mnt/data/
```

Day-to-day:

```bash
# cluster is up if this returns a Ready node
kubectl get nodes
# Argo controller/server status (namespace is OpenChoreo's build/workflow plane)
kubectl get pods -n openchoreo-workflow-plane
# controller/server logs
kubectl logs -n openchoreo-workflow-plane -l app=workflow-controller
kubectl logs -n openchoreo-workflow-plane -l app=argo-server

kubectl get workflowtemplates -n default               # confirm train-register-golden-path / fine-tune-golden-path exist
kubectl get workflows -n default                       # list runs triggered via POST /trigger-training
kubectl get workflows -n default -w                    # watch a run's phase live
kubectl logs -n default <pod-name>                     # logs of a specific train/register step pod

curl http://localhost:10081/api/v1/workflows/default    # Argo Server REST API health check (what ArgoAdapter calls, ARGO_SERVER_URL)

k3d cluster delete openchoreo-quick-start               # tear the whole cluster down (also takes out Thunder/OpenChoreo)
```

## Reference

All design decisions (golden path, tech stack, benchmarks, questions for the
mentor...) are compiled in a separate notebook — keep it alongside this repo
for reference:
[`docs/playbook-ai-delivery-portal.md`](docs/playbook-ai-delivery-portal.md)
