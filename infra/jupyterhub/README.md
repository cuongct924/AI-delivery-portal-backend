# JupyterHub (AI Notebook)

The interactive counterpart to Golden Path #1's declarative Argo training:
a Dev provisions a per-user Jupyter server, writes/runs code there, logs the
run to MLflow, then registers the model through the normal register path.

`adapters/ai_platform/notebook_adapter.py` talks to JupyterHub's own REST API
(`/hub/api/...`) — the Portal only provisions/manages; the editing surface
stays JupyterHub's web IDE.

## Deploy

```bash
helm repo add jupyterhub https://hub.jupyter.org/helm-chart/
helm upgrade --install jupyterhub jupyterhub/jupyterhub \
  -n jupyterhub --create-namespace -f infra/jupyterhub/values.yaml
```

Pinned onto worker2 (the AI Platform zone) so spawned notebook pods land next
to MLflow/MinIO and reach them over in-cluster Service DNS.

## Reach it from the host

The host-run orchestration-api and the browser both need a route in:

```bash
kubectl port-forward svc/hub          -n jupyterhub 8081:8081   # hub API
kubectl port-forward svc/proxy-public -n jupyterhub 8888:80     # notebook UI
```

Port **8888**, not 8080: `thunder.openchoreo.localhost` resolves to
`127.0.0.1`, so a forward on 8080 shadows Thunder and breaks Backstage
sign-in (the browser lands on JupyterHub's 404 instead of the IdP).

`.env` already points at both (`JUPYTERHUB_URL`, `JUPYTERHUB_PUBLIC_URL`) and
carries the service token. The token must match
`hub.services.orchestration-api.apiToken` in `values.yaml`.

## How the adapter maps to the spawner

The spawner's `profileList` (in `values.yaml`) defines one profile
(`slug: ai-notebook`) whose `profile_options` are exactly the keys the
adapter sends. Two non-obvious details, both load-bearing:

- The spawn API takes **flat** `user_options` (`profile` + one key per
  option), not a nested `profile_options` dict — the nested shape is
  silently ignored and every option falls back to its default.
- Option values are choice **keys** (strings), so the adapter stringifies
  the numeric ones.

Every key the adapter sends must exist as a `profile_options` entry, or the
spawn 400s. Adding a new `environment`/`gpu_type`/etc. means editing both
`values.yaml` and the adapter's `_spawn`.

## Known gaps

- `JupyterHubAdapter._specs` (the resource fields) is in-process only —
  JupyterHub's user API doesn't echo the spawner profile back, so a
  `list_notebooks` after an API restart shows empty resource fields until the
  next create.
- The `pytorch-cuda`/`tensorflow-cuda` profiles use CPU images (no GPU on the
  k3d nodes) — the `gpu_type` option still requests `nvidia.com/gpu`, which
  only schedules if a GPU node is present.
