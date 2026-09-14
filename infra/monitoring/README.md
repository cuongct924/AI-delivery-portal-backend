# monitoring

Configuration for tracking model/system health — Prometheus scraping +
Grafana dashboards.

- `prometheus.yml` — scrape config that **works right away** via
  `docker compose up` (`prometheus` service, UI at `http://localhost:9090`).
  `agents/mcp-servers/observability-server/` reads data from here to
  answer Claude when asked whether a model is drifting or slow.
- `grafana/` — **works right away** via `docker compose up` (`grafana`
  service, UI at `http://localhost:3001` — host port 3001, since Backstage's
  dev server already owns 3000). Auto-provisioned on startup, no manual
  setup: `grafana/provisioning/datasources/prometheus.yml` wires up
  Prometheus as the default datasource, `grafana/provisioning/dashboards/`
  auto-loads any dashboard JSON dropped into `grafana/dashboards/`.
  `grafana/dashboards/orchestration-api.json` is a starter dashboard
  (request rate, p95 latency, total requests) built from the metrics
  `services/orchestration-api/main.py` already exposes at `/metrics`.
  Anonymous viewer access is enabled for convenience; login is `admin`/`admin`
  if you need to edit anything.
- ServiceMonitor (CRD for Prometheus Operator on a real K8s cluster) —
  not implemented yet, once a real cluster exists.

## What's real vs. what's a known gap

- **Real today**: only `orchestration-api`'s own HTTP metrics (request rate,
  p95 latency, total requests, all from `prometheus_fastapi_instrumentator`)
  — scraped, dashboarded, and surfaced on the Catalog entity page via the
  `prometheus.io/rule` and `grafana/overview-dashboard` annotations on
  `Component: orchestration-api` (`examples/entities.yaml`).
- **Not wired yet, and why**: model-level/KServe inference metrics
  (`model_inference_duration_ms_bucket`, used by
  `observability-server`'s `check_model_latency` tool). This is
  **not** because KServe/BentoML don't expose `/metrics` — they do. The
  real blocker is a network gap: this docker-compose `prometheus` container
  and the local k3d `openchoreo-quick-start` cluster sit on separate Docker
  networks with no bridge between them, and its LoadBalancer's port mappings
  (see `infra/openchoreo/README.md`) cover the OpenChoreo/Thunder/Argo
  surfaces — nothing for a KServe predictor's metrics port. Closing this
  gap needs a real network bridge (or running Prometheus inside the
  cluster), not a Prometheus/KServe config change — out of scope until a
  real (non-local) cluster exists.
