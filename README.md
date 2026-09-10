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
[`cuongct924/backstage-plugins`](https://github.com/cuongct924/backstage-plugins).

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
├── docker-compose.yml    ← full local stack (mlflow, keycloak, prometheus, grafana, qdrant, minio, litellm, orchestration-api, MCP servers)
└── README.md
```

Portal frontend (Golden Path Scaffolder templates, Catalog entities, custom
Scaffolder actions/fields) lives entirely in the separate frontend repo —
not here.

See [`docs/playbook-ai-delivery-portal.md`](docs/playbook-ai-delivery-portal.md) for the full component breakdown and design rationale.

## Usage

```bash
cp .env.example .env            # fill in ANTHROPIC_API_KEY before running orchestration-api/litellm
docker compose --profile mlops up -d       # MLOps only: mlflow, keycloak, minio, orchestration-api
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
  [`cuongct924/backstage-plugins`](https://github.com/cuongct924/backstage-plugins)
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

[`scripts/setup-k8s-local.sh`](scripts/setup-k8s-local.sh) creates the `kind` cluster + Argo Workflows that back Golden Path #1. Once it's up:

```bash
# switch kubectl back if it points elsewhere
kubectl config use-context kind-ai-delivery-portal

# cluster is up if this returns a Ready node
kubectl get nodes
# Argo controller/server status
kubectl get pods -n argo
# debug a stuck/crashing pod                              
kubectl describe pod -n argo -l app=workflow-controller
# controller logs
kubectl logs -n argo -l app=workflow-controller
# server logs
kubectl logs -n argo -l app=argo-server                

kubectl get workflowtemplates -n default               # confirm train-register-golden-path / fine-tune-golden-path exist
kubectl get workflows -n default                       # list runs triggered via POST /trigger-training
kubectl get workflows -n default -w                    # watch a run's phase live
kubectl logs -n default <pod-name>                     # logs of a specific train/register step pod

curl http://localhost:2746/api/v1/workflows/default    # Argo Server REST API health check (what ArgoAdapter calls)

kind delete cluster --name ai-delivery-portal          # tear the whole cluster down
```

## Reference

All design decisions (golden path, tech stack, benchmarks, questions for the
mentor...) are compiled in a separate notebook — keep it alongside this repo
for reference:
[`docs/playbook-ai-delivery-portal.md`](docs/playbook-ai-delivery-portal.md)
