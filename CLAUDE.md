# AI Delivery Portal (backend)

Orchestration API (FastAPI) + AI Agent/MCP + Adapter layer + GitOps infra.
This repo is the **backend** half of AI Delivery Portal. The Portal frontend
(Backstage-based, OpenChoreo plugins + Viettel Cloud branding) lives in a
**separate repo**: [`cuongct924/backstage-plugins`](https://github.com/cuongct924/backstage-plugins)
(fork of `openchoreo/backstage-plugins`), local clone at
`/Users/cuongct090_04/Code/backstage-plugins`. See
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

Local infra:
```bash
docker compose up -d   # mlflow, keycloak, prometheus, grafana, qdrant, minio, litellm, orchestration-api, 3 MCP servers
bash scripts/run-mcp-local.sh observability|golden-paths   # run one MCP server without Docker
```

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
  frontend repo (`cuongct924/backstage-plugins`), not here.
