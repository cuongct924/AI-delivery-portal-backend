# observability-server

MCP server for the "system/model health" domain — read-only, merges what
used to be 3 separate servers (`mlops-server`, `k8s-server`, `metrics-server`).

Tools — MLOps side:
- `list_experiments`, `get_model_metrics` — reads MLflow
  (`adapters/mlflow_adapter.py`), set up via `docker compose up mlflow`.
- `check_pod_status`, `get_logs` — currently **mock** (not wired to a real
  K8s cluster). Integrate the `kubernetes` Python client when a real cluster
  is available, and add RBAC/OPA policy in `infra/opa-policies/` before
  granting write access.
- `query_metric` (free-form PromQL), `check_model_latency` (now requires
  `namespace` — multi-env x multi-tenant means the same model can run in
  several namespaces at once, see `infra/environments/README.md`) — reads
  Prometheus (set up via `docker compose up`, scrape config at
  `infra/monitoring/prometheus.yml`).
- `get_promotion_status` — currently **mock**: the originally-planned
  Kargo-based promotion pipeline was removed early on and never actually
  installed; `infra/openchoreo/deployment-pipeline.yaml`'s
  `DeploymentPipeline` is the current intended replacement, not yet wired
  (see `docs/openchoreo-migration-next-steps.md` Phase 3/4). Read-only by
  design: approving a prod promotion stays a human action, this tool only
  reports state, never triggers one.

Tools — LLMOps side:
- `get_llm_spend` — real, reads LiteLLM's own spend ledger (`GET
  /global/spend/report` via `adapters/llm_gateway_adapter.py`'s
  `get_spend_report`), set up via `docker compose up litellm`.
- `get_active_prompt_version`, `get_active_rag_version` — real, call
  orchestration-api's `GET /prompts` / `GET /rag/{collection}` (not a
  shared file — see this server's `server.py` header comment for why).
  LLMOps releases are Instant-only (`docs/llmops-lifecycle-plan.md` mục
  Q4), so there's no Git/ArgoCD trail to read this from otherwise.
- `get_eval_score_trend` — currently **mock**: `judge_response()`
  (`evaluations/llm_judge.py`) scores each call but nothing persists the
  result anywhere yet.

Transport: `streamable-http` — discovered via the Backstage Catalog, not a
local subprocess. `MCP_HOST`/`MCP_PORT` (default `0.0.0.0:9001`).

## Run locally

```bash
bash scripts/run-mcp-local.sh observability
```
