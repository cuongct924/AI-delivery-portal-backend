# environments

> **Superseded by OpenChoreo (Phase 2, sub-step 2.2)** — the ArgoCD/Kargo/
> Kubara pieces this doc describes below (`infra/argocd/`, `infra/kargo/`,
> `infra/kubara/`) were deleted; `Environment`/`DeploymentPipeline` in
> `infra/openchoreo/` replace their role (see that directory's own
> README). This directory itself still holds the real PR-gated
> `inference-services/`/`orchestration-api`/`portal` overlay values Golden
> Path #2 writes to — not yet migrated to `infra/openchoreo/` — so the rest
> of this doc is kept as historical context for that part only.

Per-environment GitOps targets — replaces the old flat
`infra/inference-services/` (single, env-less directory).

```
dev/inference-services/{mlops-team,llmops-team}/<model>/<version|llm>.yaml
dev/orchestration-api/values.yaml
dev/portal/values.yaml
staging/...   # same shape
prod/...      # same shape
```

`inference-services/` is further partitioned by **tenant**
(`mlops-team`/`llmops-team`) — multi-environment × multi-tenant, per
ArgoCD's [env] × [tenant] `AppProject` convention (see
`infra/argocd/README.md`). `orchestration-api/`/`portal/` are not
multi-tenant resources, so they stay 1-per-environment only.

Each `(env, tenant)` pair for `inference-services/` maps to its own
Kubernetes namespace (`ai-delivery-portal-<env>-<tenant>`); each `env` for
`orchestration-api`/`portal` maps to `ai-delivery-portal-<env>` — all on
the same local `kind` cluster, see `infra/k8s-local-cluster/`. No real
multi-cluster infra is required; `infra/argocd/applicationset-*.yaml`'s
`git`/`list` generators can be extended to a `cluster` generator later if
real multi-cluster infra shows up.

**Who writes what — this is the important part:**

- **`dev/inference-services/<tenant>/`** — written directly by Backstage
  Golden Paths (`register-deploy` → `mlops-team`, `deploy-llm` →
  `llmops-team`) via `orchestration-api`'s `prepare_deploy_manifest`
  (`services/orchestration-api/routers/models.py`), PR-gated: nothing here
  is meant to be hand-edited, each file's lifecycle is PR-driven, deleting
  a file removes the matching `InferenceService` (`prune: true`).
- **`staging/` and `prod/` `inference-services/<tenant>/`** — intended to
  be written **only by that tenant's own promotion pipeline**, never by
  Backstage, never by hand, and never across tenants: `staging` promotes
  automatically, `prod` only after manual approval, per tenant,
  independently. The originally-planned mechanism for this was Kargo
  (`infra/kargo/`) — removed early on, never actually installed.
  `infra/openchoreo/deployment-pipeline.yaml`'s `DeploymentPipeline` is the
  current intended replacement (same promotion-chain shape), not yet
  wired — see `docs/openchoreo-migration-next-steps.md` Phase 3/4.
- **`orchestration-api/`, `portal/` (all envs)** — not tenant-scoped, not
  Golden-Path-written; overlay values only, owned by platform-team.

Base component versions/capabilities across all 3 environments are
declared once in `infra/kubara/config.yaml` and used to generate the
per-env/per-tenant skeleton + the ArgoCD `ApplicationSet`/`AppProject`
wiring — see `infra/kubara/README.md`.
