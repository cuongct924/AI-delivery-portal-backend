#!/usr/bin/env bash
# Forwards the AI Platform zone services (worker2, ai-platform-zone namespace)
# to localhost so the host-run orchestration-api (`make run-orchestration-api`)
# can reach them — their ClusterIPs aren't routable from the host.
#
# Ports match .env's *_URL defaults (QDRANT_URL, MLFLOW_TRACKING_URI,
# LITELLM_GATEWAY_URL, DVC_S3_ENDPOINT_URL, FEAST_SERVING_URL). Run this in a
# separate terminal and leave it up; Ctrl-C tears every forward down.
#
# Self-healing: `kubectl port-forward` is a plain foreground process with no
# reconnect logic of its own — a pod restart, a laptop sleep/wake, or any
# transient network blip kills it silently and it never comes back on its
# own (this repeatedly bit the portal-assistant work: the forward would die
# and every chat/costs/security call would 61-refuse until someone noticed
# and re-ran the script by hand). Each forward below now runs in its own
# retry loop instead of a one-shot invocation.
#
# Usage: bash scripts/port-forward-ai-platform.sh [qdrant|mlflow|litellm|minio|feast-serve ...]
#        (no args = all of them)
set -euo pipefail

CTX="${KUBECTL_CONTEXT:-k3d-openchoreo-quick-start}"
NS="ai-platform-zone"

# name:local:remote
ALL_FORWARDS=(
  "qdrant:6333:6333"
  "mlflow:5000:5000"
  "litellm:4000:4000"
  "minio:9000:9000"
  "feast-serve:6566:6566"
)

if [ "$#" -gt 0 ]; then
  FORWARDS=()
  for want in "$@"; do
    match=""
    for f in "${ALL_FORWARDS[@]}"; do
      [ "${f%%:*}" = "$want" ] && match="$f"
    done
    if [ -z "$match" ]; then
      echo "Unrecognized service: $want (qdrant|mlflow|litellm|minio|feast-serve)" >&2
      exit 1
    fi
    FORWARDS+=("$match")
  done
else
  FORWARDS=("${ALL_FORWARDS[@]}")
fi

# Restarts `kubectl port-forward` for one service forever, with a short
# backoff, until this subshell is signaled (the outer trap below sends TERM
# to it on Ctrl-C). kubectl's own stdout/stderr is silenced per-attempt
# (just "Forwarding from ..." on success, "error: ..." on the connection it
# lost) — only our own retry message prints, so a flapping connection
# doesn't spam the terminal. The inner trap kills the *current* kubectl
# child before exiting so Ctrl-C doesn't leave an orphaned forward running.
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

echo "Forwards up (context=$CTX) and self-healing. Ctrl-C to stop."
wait
