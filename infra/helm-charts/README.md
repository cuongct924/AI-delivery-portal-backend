# helm-charts

Helm chart per service, each at `infra/helm-charts/<service-name>/` with
`Chart.yaml`, `values.yaml`, `templates/` — implemented for exactly 2:

- **`orchestration-api/`**
- **`portal/`** (Backstage backend — runs with `app-config.yaml` dev-mode
  config, in-memory SQLite + guest auth, not `app-config.production.yaml`,
  which needs a Postgres that doesn't exist anywhere in this project yet)

Previously synced onto the local `kind` cluster via ArgoCD
(`applicationset-orchestration-api.yaml`/`applicationset-portal.yaml`) —
that ArgoCD wiring (`infra/argocd/`) was deleted in Phase 2's OpenChoreo
migration (sub-step 2.2); see `infra/openchoreo/platform/` for the
Component/Workload manifests that now deploy these 2 services instead
(`ComponentType` "service"/"web-application", real image builds proven
end to end). **Additive to `docker compose up -d`/`yarn start`, not a
replacement** — this is the GitOps/production-like verification path,
local dev keeps using docker-compose for its fast code-then-see-it loop.

**Deliberately no chart for the 3 MCP servers** (`mlops-server`/
`k8s-server`/`metrics-server`) — they're stdio-transport tools spawned
on-demand (`docker compose run -i ...`, `profiles: ["manual"]` in
`docker-compose.yml`), not long-running services. A K8s `Deployment`
would crash-loop them (no stdin attached, the process exits on EOF).
