#!/usr/bin/env bash
# Forwards the AI Platform zone services (worker2, ai-platform-zone namespace)
# to localhost so the host-run orchestration-api (`make run-orchestration-api`)
# can reach them — their ClusterIPs aren't routable from the host.
#
# Ports match .env's *_URL defaults (QDRANT_URL, MLFLOW_TRACKING_URI,
# LITELLM_GATEWAY_URL, DVC_S3_ENDPOINT_URL, FEAST_SERVING_URL). Run this in a
# separate terminal and leave it up; Ctrl-C tears every forward down.
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

pids=()
cleanup() {
  for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

for f in "${FORWARDS[@]}"; do
  IFS=: read -r svc local remote <<<"$f"
  echo "port-forward $NS/$svc  localhost:$local -> $remote"
  kubectl --context "$CTX" -n "$NS" port-forward "svc/$svc" "$local:$remote" >/dev/null 2>&1 &
  pids+=("$!")
done

echo "Forwards up (context=$CTX). Ctrl-C to stop."
wait
