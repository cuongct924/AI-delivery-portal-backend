# OpenChoreo (Phase 2)

Declarative snapshot of what's applied on the local OpenChoreo control plane
(self-hosted via the k3d quick-start — see `openchoreo.dev/docs/getting-started/try-it-out/on-k3d-locally/`),
exported by hand from `kubectl get <resource> -o yaml` and stripped of
runtime fields (`status`, `resourceVersion`, `uid`, `finalizers`,
`creationTimestamp`). Not yet applied by any automation — see
`docs/playbook-ai-delivery-portal.md` / the Phase 2 plan for status.

## Sub-step progress

- **2.0 — control plane confirmed**: real k3d cluster
  (`k3d-openchoreo-quick-start-*` containers), all `openchoreo.dev` CRDs
  present, `Project`/`Environment`/`DeploymentPipeline` already proven live
  by `fraud-detection/` before this session touched anything.
- **2.1 — `orchestration-api` as a real Component**: done, see
  `platform/component-orchestration-api.yaml` +
  `workload-orchestration-api.yaml`. Confirmed end to end — `200` from
  `http://development-default.openchoreoapis.localhost:19080/orchestration-api-http/openapi.json`,
  the real FastAPI app, not a placeholder. Getting there surfaced (and
  fixed) 4 real, previously-latent bugs — this image had apparently never
  actually been built+run before:
  - `services/orchestration-api/Dockerfile`: the `uvicorn` console-script
    installed by `uv pip install --target` landed off `$PATH`
    (`site-packages/bin/uvicorn`) — fixed via `ENV PATH`.
  - Same Dockerfile never installed `adapters/`'s own
    `requirements.lock.txt` despite vendoring its source — fixed by
    installing both lock files together in one `uv pip install` call
    (surfaced 9 real version conflicts between the 2 independently-locked
    files, resolved via `container-overrides.txt` + uv's `--overrides`).
  - Same Dockerfile never copied `infra/feature-store/` in, but
    `FeastAdapter()` is constructed eagerly at import time
    (`routers/models.py` module scope) and needs it — fixed via `COPY`.
  - MLflow 3.x's DNS-rebinding protection 403s any request whose `Host`
    header isn't localhost — every in-cluster caller reaching it via
    `host.docker.internal` was rejected. Fixed via `MLFLOW_SERVER_ALLOWED_HOSTS`
    in `docker-compose.yml` (server config, not a client bug).
  - `portal/`'s own image is built from `packages/backend/Dockerfile` in
    the frontend repo — declared (`component-portal.yaml` +
    `workload-portal.yaml`) but deploy is a separate, later pass; that
    Dockerfile installs `app-config.production.yaml` (OpenChoreo upstream's
    own config surface — `OPENCHOREO_API_URL`/`THUNDER_BASE_URL`/etc, not
    this fork's dev `.env`), which has no MLOps template
    `catalog.locations` entries at all. The env vars in
    `workload-portal.yaml` were copied from the already-running stock
    `backstage` Deployment in `openchoreo-control-plane` (proven working
    in this same cluster) and simplified to guest mode.
- **2.2 — `argocd`/`kargo`/`kubara` deleted**: done. `Environment`
  (`environments.yaml`) + `DeploymentPipeline` (`deployment-pipeline.yaml`)
  already existed from the `fraud-detection/` spike and needed no new work
  — both `platform` and `fraud-detection` Projects reuse the same 3
  Environments and the same `default` DeploymentPipeline. Every remaining
  code comment that pointed at `infra/kargo/README.md` (`adapters/factory.py`,
  `routers/models.py`, `routers/llm_serving.py`) now points here instead;
  `scripts/setup-kserve-argocd-local.sh` (installed ArgoCD, applied
  `infra/argocd/`) was deleted as dead weight.
- **2.3 — not started for real**: KServe isn't installed on this cluster
  at all (no CRDs — confirmed via `kubectl get crds`), so an
  `InferenceService` `ClusterComponentType` has nothing to wrap yet; that's
  a real prerequisite, not just unwritten YAML. A `ClusterWorkflowPlane`
  (`default`) does already exist, so wrapping Golden Path #1's training
  logic in a real `Workflow`/`ClusterWorkflow` is achievable without new
  infra — just not done yet. Migrating Keycloak → Thunder needs a real
  user/role list built from `infra/keycloak/realm-export.json` first (see
  the Phase 2 plan's own mục 6.2) — a manual step, not something to script
  blind.

## Known OpenChoreo control-plane bug hit live in this session

`controller-manager`'s `DeploymentPipeline`/`ReleaseBinding→RenderedRelease`
reconcilers repeatedly got stuck — a brand-new `Project`'s
`ProjectReleaseBinding` never got created despite the Project reconciling
fine and "enqueueing DeploymentPipeline" logging every time; separately, a
`Component` parameter change (CPU/memory bump) produced a new
`ComponentRelease` correctly but the actual `Deployment` in the data-plane
namespace never picked it up. Both were worked around by restarting
`controller-manager` and, once, by patching the `Deployment` object
directly — real gaps in this self-hosted install's reconciliation
reliability, not application bugs. Worth flagging before relying on this
control plane for anything more than local spikes.

## What this proves

`fraud-detection/` is a spike validating that OpenChoreo's `Component`
(`cronjob/scheduled-task`) + `Environment` + `DeploymentPipeline` model can
replace Argo CronWorkflow/ArgoCD/Kargo's role as the **scheduling/promotion
layer**, without the real MLOps domain logic (`adapters/`,
`services/orchestration-api`) changing at all:

- `component-training.yaml` + `workload-training.yaml` — a CronJob that
  calls `POST http://host.docker.internal:8000/trigger-training` on the
  real orchestration-api (reachable from inside the k3d cluster) every 3
  minutes. Verified end to end: the CronJob's Job triggers
  `scripts/local-demo/fake_argo.py` (Golden Path #1's stand-in for a `kind`
  cluster + Argo Workflows), which trains a real model on
  `data/traditional-ml/fraud-detection-sample.csv` and registers it as `fraud-detection`
  in the real local MLflow via `/models/register` — same as any other
  Golden Path #1 caller.
- `component-serving.yaml` + `workload-serving.yaml` — currently a
  placeholder (`nginxdemos/hello`), **not yet wired** to the registered
  model or to `orchestration-api`/KServe. Blocked on 2.3's KServe
  `ComponentType`, per above.
- `project.yaml`, `environments.yaml`, `deployment-pipeline.yaml` — the
  Organization-equivalent boundary, the 3 environments, and the
  dev→staging→prod promotion path (`DeploymentPipeline`), replacing
  the [env]×[tenant] AppProject matrix / Kargo Stages respectively.

`platform/` proves the same model works for the Portal's own 2 services
(`orchestration-api` as `deployment/service`, `portal` as
`deployment/web-application`) — see the 2.1 section above.

## What's NOT done yet

- `fraud-detection-serving` isn't wired to a real model — no KServe
  `ComponentType`/Trait exists yet for "InferenceService" (2.3, blocked on
  KServe not being installed).
- No `Resource`/`ResourceType` wraps MLflow/Qdrant/LiteLLM/Feast — they
  keep running as plain `docker-compose.yml` services, reached from inside
  the cluster via `host.docker.internal`.
- Promotion (dev→staging via `DeploymentPipeline`) hasn't been exercised —
  only the `development` environment has been triggered so far.
- Thunder (bundled with this same OpenChoreo install, namespace `thunder`)
  hasn't replaced Keycloak yet — `services/orchestration-api/auth/keycloak.py`
  is unchanged. `portal/`'s own Workload runs in guest mode (no real
  Thunder OAuth round-trip) for the same reason.
- `portal/`'s image is declared but not yet built+deployed (separate pass
  from `orchestration-api`'s).
