# AI Delivery Portal (backend)

Orchestration API (FastAPI) + AI Agent/MCP + Adapter layer + GitOps infra.
This repo is the **backend** half of AI Delivery Portal. The Portal frontend
(Backstage-based, OpenChoreo plugins + Viettel Cloud branding) lives in a
**separate repo**: [`cuongct924/AI-delivery-portal-frontend`](https://github.com/cuongct924/AI-delivery-portal-frontend)
(fork of `openchoreo/backstage-plugins`, renamed after forking), local clone at
`/Users/cuongct090_04/Code/backstage-plugins` (directory name predates the
GitHub rename — not yet renamed locally). See
`docs/playbook-ai-delivery-portal.md` for the component diagram and design
decisions/rationale.

## Commands

Python (each service under `adapters/`, `services/orchestration-api/`,
`agents/mcp-servers/*` has its own `requirements.txt` — see Makefile
`SERVICE_REQS`):
```bash
make install   # create .venv, install dev.lock.txt + pre-commit hook
make lock       # regenerate all *.lock.txt — run after editing any requirements.txt
make check      # lint + format-check + typecheck + test (what CI runs)
make run-orchestration-api / run-observability-mcp / run-golden-paths-mcp
```

`make test`/`make check` need a real MLflow reachable at `MLFLOW_TRACKING_URI`
(default `http://localhost:5000`) — `routers/prompts.py` seeds its default
personas via a live Prompt Registry at *import* time, before any test can
mock it. CI starts a throwaway container for this (`.github/workflows/ci.yml`);
do the same locally before running tests (the 3-node cluster's MLflow, on
worker2, isn't reachable at `localhost:5000`):
```bash
docker run -d --name mlflow-test -p 5000:5000 ghcr.io/mlflow/mlflow:v3.15.1 \
  mlflow server --host 0.0.0.0 --port 5000
```

Local infra runs on a 3-node k3d cluster (`k3d-openchoreo-quick-start`), not
docker-compose (retired):
```bash
bash scripts/setup-3node-infra.sh   # idempotent — adds/labels/taints worker1+worker2, builds+imports images, applies manifests
bash scripts/run-mcp-local.sh observability|golden-paths   # run one MCP server without the cluster
```
Node layout — **master** (`k3d-openchoreo-quick-start-server-0`): OpenChoreo
control plane (API, controllers, Thunder); **worker1**
(`k3d-worker1-0`, `plane.viettel.vn=data-plane-portal`): `orchestration-api`
+ adapter layer + the MCP servers, as OpenChoreo Components
(`infra/openchoreo/namespaces/default/projects/platform/`); **worker2**
(`k3d-worker2-0`, `plane.viettel.vn=ai-platform-workflow`): the
training/fine-tune workflow plane (Argo) **and** the AI Platform zone —
MLflow/Qdrant/MinIO/LiteLLM/Feast, plain k8s manifests outside OpenChoreo
(`infra/ai-platform-zone/`), representing Viettel's own AI Platform SPDV.

`infra/openchoreo/` is split by scope: `platform-shared/` holds
cluster-scoped resources (ClusterComponentType, ClusterResourceType,
ClusterAuthzRole(Binding), ClusterWorkflow, plus the cluster-scoped Argo
`ClusterWorkflowTemplate`); `namespaces/default/` holds namespace-scoped
Environment/DeploymentPipeline (`platform/`) and per-project
Project/Component/Workload/ReleaseBinding/Resource
(`projects/<name>/`, each component under its own `components/<name>/`).
Kubernetes `ClusterRole`/`ClusterRoleBinding` (not OpenChoreo resources)
live in `infra/bootstrap/`.

Run `make check` before committing — CI (`.github/workflows/ci.yml`) runs the
exact same commands, nothing else.

## Coding standards

- Python: see `.claude/rules/python-standards.md` (ruff + pyright + pytest,
  Google-style docstrings, class layout order). Applies to `adapters/`,
  `agents/`, `services/orchestration-api/`.
- Every Adapter implements a shared interface from `adapters/interfaces.py`
  (Adapter Pattern) — switching Mock → real backend (MLflow/KServe/Argo/...)
  means adding one new class, never touching callers.
- **Business logic never lives in the Portal frontend.** A Custom Scaffolder
  Action (in the frontend repo) only makes an HTTP call to
  `services/orchestration-api`; the FastAPI service here owns
  Adapter/Factory/policy logic.

## Architecture (directories)

```
services/             orchestration-api — FastAPI BFF, MCP client, auth, evaluations
agents/               AI Agent & MCP — mcp-servers/, skills/
adapters/             Adapter Pattern — MLflow, KServe, Argo, Qdrant, LiteLLM, Feast, JupyterHub
data/                 DVC-tracked datasets — pointer files in git, real data in an S3-compatible remote
infra/                GitOps infra — monitoring/vector-dbs/llm-gateways (active); helm-charts/argocd/opa-policies (not yet implemented)
docs/                 playbook, LLMOps draft plan
```

Portal frontend (Golden Path Scaffolder templates, Catalog entities, custom
Scaffolder actions/fields) lives entirely in the separate frontend repo —
not here.

## Workflow

- Repo is hosted on **GitHub**; CI runs via `.github/workflows/ci.yml`.
- Adding a new Python service: add its `requirements.txt` path to `Makefile`'s
  `SERVICE_REQS`, run `make lock`, add it to `docker-compose.yml`, and add a
  build+scan block to `ci.yml`.
- Adding/changing a Golden Path template or Catalog entity: do it in the
  frontend repo (`cuongct924/AI-delivery-portal-frontend`), not here.
