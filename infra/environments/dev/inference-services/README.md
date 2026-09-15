# dev/inference-services

Partitioned by tenant — `mlops-team/` and `llmops-team/` — each a
completely isolated ArgoCD sync target (see
`infra/argocd/appproject-dev-mlops-team.yaml`/
`appproject-dev-llmops-team.yaml` and
`infra/argocd/applicationset-inference-services-dev.yaml`). A tenant's own
deploys never touch the other tenant's `AppProject`/namespace, even though
both are `dev`.

- `mlops-team/<model>/<version>.yaml` — written by Golden Path #2
  (`register-deploy`, MLflow-registered models).
- `llmops-team/<model>/llm.yaml` — written by "Serving LLM"
  (`deploy-llm`, self-hosted LLMs via vLLM).

Both via `services/orchestration-api/templates/inference_service.yaml.j2`
(`routers/models.py`'s `prepare_deploy_manifest`), PR-gated. Nothing here
is hand-edited; each file's lifecycle is entirely PR-driven, and deleting
one removes the matching `InferenceService` (`prune: true`).

This is the **only** tier Backstage writes to directly.
`../../staging/inference-services/<tenant>/` and
`../../prod/inference-services/<tenant>/` are intended to be populated
exclusively by that tenant's own promotion pipeline out of here — the
originally-planned Kargo `Warehouse`/`Stage` pair for this was removed
early on and never actually installed; see `../../README.md` for the
current OpenChoreo `DeploymentPipeline` replacement (not yet wired).
