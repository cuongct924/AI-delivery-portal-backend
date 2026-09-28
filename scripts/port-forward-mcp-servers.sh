#!/usr/bin/env bash
# Forwards the platform project's MCP servers (worker1, the OpenChoreo-managed
# dp-default-platform-development-* namespace) to localhost so the host-run
# orchestration-api (`make run-orchestration-api`) can reach them — their
# ClusterIPs aren't routable from the host, and its MCP discovery
# (catalog_client.discover_mcp_servers) resolves endpoints straight from the
# Backstage Catalog, which points at each server's in-cluster Service *name*
# (e.g. http://golden-path-guide-server:9003/mcp), not localhost.
#
# A port-forward alone isn't enough — the host also needs each Service name
# to resolve to 127.0.0.1. Add these once to /etc/hosts:
#   127.0.0.1 golden-path-guide-server
#   127.0.0.1 llmops-golden-paths-server
#   127.0.0.1 mlops-golden-paths-server
#   127.0.0.1 observability-server
#
# Run this in a separate terminal and leave it up; Ctrl-C tears every
# forward down.
#
# Self-healing: each forward runs in its own auto-reconnect loop (see
# supervise_forward below) — a pod restart or network blip no longer leaves
# it silently dead until someone notices and re-runs this script by hand.
#
# Usage: bash scripts/port-forward-mcp-servers.sh [golden-path-guide|llmops-golden-paths|mlops-golden-paths|observability ...]
#        (no args = all of them)
set -euo pipefail

CTX="${KUBECTL_CONTEXT:-k3d-openchoreo-quick-start}"

NS="$(kubectl --context "$CTX" get ns \
  -l 'openchoreo.dev/project=platform,openchoreo.dev/environment=development' \
  -o jsonpath='{.items[0].metadata.name}')"
if [ -z "$NS" ]; then
  echo "Could not resolve the platform/development dataplane namespace — is scripts/setup-3node-infra.sh applied?" >&2
  exit 1
fi

# name:local:remote — name is both the kubectl Service name and the
# hostname it must resolve as (see the /etc/hosts note above).
ALL_FORWARDS=(
  "golden-path-guide-server:9003:9003"
  "llmops-golden-paths-server:9002:9002"
  "mlops-golden-paths-server:9004:9004"
  "observability-server:9001:9001"
)

if [ "$#" -gt 0 ]; then
  FORWARDS=()
  for want in "$@"; do
    match=""
    for f in "${ALL_FORWARDS[@]}"; do
      case "${f%%:*}" in
        "$want"|"$want-server") match="$f" ;;
      esac
    done
    if [ -z "$match" ]; then
      echo "Unrecognized service: $want (golden-path-guide|llmops-golden-paths|mlops-golden-paths|observability)" >&2
      exit 1
    fi
    FORWARDS+=("$match")
  done
else
  FORWARDS=("${ALL_FORWARDS[@]}")
fi

# See port-forward-ai-platform.sh's copy of this function for the full
# rationale on the retry loop + inner trap.
supervise_forward() {
  local svc="$1" local_port="$2" remote_port="$3"
  local cur_pid=""
  trap '[ -n "$cur_pid" ] && kill "$cur_pid" 2>/dev/null; exit 0' TERM INT
  while true; do
    kubectl --context "$CTX" -n "$NS" port-forward "svc/$svc" "$local_port:$remote_port" >/dev/null 2>&1 &
    cur_pid=$!
    # `|| true`: kubectl dying (incl. killed by a signal) makes `wait`
    # return non-zero — under `set -e` that would abort this whole retry
    # loop right here instead of looping, which is exactly the "silently
    # stays dead" failure mode this script exists to fix.
    wait "$cur_pid" || true
    echo "[$(date +%H:%M:%S)] $NS/$svc port-forward exited — reconnecting in 2s"
    sleep 2
  done
}

pids=()
cleanup() {
  for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

for f in "${FORWARDS[@]}"; do
  IFS=: read -r svc local remote <<<"$f"
  echo "port-forward $NS/$svc  localhost:$local -> $remote (auto-reconnect)"
  supervise_forward "$svc" "$local" "$remote" &
  pids+=("$!")
done

echo "Forwards up (context=$CTX, namespace=$NS) and self-healing. Ctrl-C to stop."
echo "Reminder: each Service name above must also resolve to 127.0.0.1 in /etc/hosts."
wait
