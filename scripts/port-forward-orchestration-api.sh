#!/usr/bin/env bash
# Forwards the in-cluster orchestration-api (worker1, the OpenChoreo-managed
# dp-default-platform-development-* namespace) to localhost:8000 — the URL
# the frontend repo's Backstage is configured with (its .env
# OPENCHOREO_PORTAL_ASSISTANT_URL, app-config.local.yaml's scaffolder-action
# baseUrl and /orchestration-api proxy).
#
# This is the alternative to `make run-orchestration-api`: the API runs
# in-cluster and reaches MLflow/LiteLLM/Qdrant/MCP servers over cluster DNS,
# so no other port-forward and no /etc/hosts entries are needed on the host.
# Code changes only take effect after rebuilding + importing its image (see
# scripts/setup-3node-infra.sh step 3/6) and restarting the pod. Don't run
# both this and `make run-orchestration-api` — they both want port 8000.
#
# Self-healing, same as port-forward-mcp-servers.sh: a pod restart or network
# blip reconnects on its own instead of leaving the forward silently dead.
# Run in its own terminal and leave it up; Ctrl-C stops it.
set -euo pipefail

CTX="${KUBECTL_CONTEXT:-k3d-openchoreo-quick-start}"
LOCAL_PORT="${ORCHESTRATION_API_LOCAL_PORT:-8000}"

NS="$(kubectl --context "$CTX" get ns \
  -l 'openchoreo.dev/project=platform,openchoreo.dev/environment=development' \
  -o jsonpath='{.items[0].metadata.name}')"
if [ -z "$NS" ]; then
  echo "Could not resolve the platform/development dataplane namespace — is scripts/setup-3node-infra.sh applied?" >&2
  exit 1
fi

cur_pid=""
trap '[ -n "$cur_pid" ] && kill "$cur_pid" 2>/dev/null; exit 0' TERM INT

echo "port-forward $NS/orchestration-api  localhost:$LOCAL_PORT -> 8000 (auto-reconnect, Ctrl-C to stop)"
while true; do
  kubectl --context "$CTX" -n "$NS" port-forward svc/orchestration-api "$LOCAL_PORT:8000" >/dev/null 2>&1 &
  cur_pid=$!
  # `|| true`: under `set -e`, kubectl exiting non-zero would otherwise end
  # this retry loop instead of reconnecting.
  wait "$cur_pid" || true
  echo "[$(date +%H:%M:%S)] orchestration-api port-forward exited — reconnecting in 2s"
  sleep 2
done
