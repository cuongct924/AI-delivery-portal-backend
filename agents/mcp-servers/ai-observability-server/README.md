# ai-observability-server

MCP server for the AI/ML/LLM health domain — read-only. Covers only what
OpenChoreo has no concept of; infra-level tools (k8s pod/logs, raw PromQL,
OpenChoreo promotion state) are left to OpenChoreo's own Control Plane /
Observability Plane MCP servers.

Tools — MLOps side:
- `list_experiments`, `get_model_metrics` — reads MLflow
  (`adapters/ai_platform/mlflow_adapter.py`), set up via `docker compose up mlflow`.
- `check_model_latency` — reads Prometheus (OpenChoreo's in-cluster
  observability plane; the old compose-era `infra/monitoring/` config is
  gone). `namespace` is
  required — multi-env x multi-tenant means the same model can run in
  several namespaces at once, see `docs/openchoreo-environments.md`.

Tools — LLMOps side:
- `get_llm_spend` — real, reads LiteLLM's own spend ledger (`GET
  /global/spend/report` via `adapters/ai_platform/llm_gateway_adapter.py`'s
  `get_spend_report`), set up via `docker compose up litellm`.
- `get_active_prompt_version`, `get_active_rag_version` — real, call
  orchestration-api's `GET /prompts` / `GET /rag/{collection}` (not a
  shared file — see this server's `server.py` header comment for why).
  LLMOps releases are Instant-only (`docs/llmops-lifecycle-plan.md` mục
  Q4), so there's no Git/ArgoCD trail to read this from otherwise.
- `get_eval_score_trend` — currently **mock**: `judge_response()`
  (`evaluations/llm_judge.py`) scores each call but nothing persists the
  result anywhere yet.

Delegated to OpenChoreo's MCP servers (removed from here):
- `check_pod_status`, `get_logs` → Control Plane MCP `get_resource_logs` /
  `get_resource_events`.
- `query_metric` (raw PromQL) → Observability Plane MCP
  `query_http_metrics` / `query_resource_metrics`.
- `get_promotion_status` → Control Plane MCP `get_release_binding` /
  `list_deployment_pipelines`.

Transport: `streamable-http` — discovered via the Backstage Catalog, not a
local subprocess. `MCP_HOST`/`MCP_PORT` (default `0.0.0.0:9001`).

## Run locally

```bash
bash scripts/run-mcp-local.sh ai-observability
```