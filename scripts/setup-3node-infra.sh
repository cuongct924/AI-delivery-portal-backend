#!/usr/bin/env bash
# Stands up the 3-node k3d topology described in CLAUDE.md's "Local infra"
# section: adds worker1 ("data plane portal") and worker2 ("workflow
# plane" + "AI Platform zone") to the existing openchoreo-quick-start
# cluster, seeds their hostPath mounts, builds/imports every local image,
# and applies the AI-platform-zone + OpenChoreo platform manifests.
#
# Idempotent — safe to re-run after editing any manifest or Dockerfile.
# Does NOT touch the cluster's existing control-plane install (OpenChoreo,
# Thunder, KServe, cert-manager, ...) — see CLAUDE.md and the plan this
# script came from for why (that bootstrap isn't reproducible from this
# repo alone).
#
# Usage: bash scripts/setup-3node-infra.sh
#
# Verbose by default — every command this script runs is echoed (with a
# timestamp) as it executes, not just its own step banners, and the full
# session is also saved to scripts/.setup-logs/ for later review.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/scripts/.setup-logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup-3node-infra-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1

CLUSTER=openchoreo-quick-start
CTX="k3d-${CLUSTER}"
WORKER1=k3d-worker1-0
WORKER2=k3d-worker2-0

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

export PS4='+ [$(date -u +%H:%M:%S)] '
set -x

log "1/6 — node topology (worker1=data-plane-portal, worker2=ai-platform-workflow)"
if ! k3d node list worker1-0 >/dev/null 2>&1; then
  k3d node create worker1 --cluster "$CLUSTER" --role agent \
    --k3s-node-label plane.viettel.vn=data-plane-portal --wait
fi
if ! k3d node list worker2-0 >/dev/null 2>&1; then
  k3d node create worker2 --cluster "$CLUSTER" --role agent \
    --k3s-node-label plane.viettel.vn=ai-platform-workflow --wait
fi

kubectl --context "$CTX" label node "k3d-${CLUSTER}-server-0" plane.viettel.vn=control-plane --overwrite
kubectl --context "$CTX" taint node "$WORKER1" dedicated.viettel.vn=data-plane-portal:NoSchedule --overwrite
kubectl --context "$CTX" taint node "$WORKER2" dedicated.viettel.vn=ai-platform-workflow:NoSchedule --overwrite

log "2/6 — seeding hostPath data (docker cp — re-run any time ./data or infra/feature-store/data changes)"
# worker2 needs /mnt/data too: the training ClusterWorkflowTemplate
# (infra/openchoreo/platform-shared/cluster-workflow-templates/argo/train-register-cluster-template.yaml)
# is now pinned
# there and mounts the same hostPath orchestration-api uses on worker1.
docker exec "$WORKER2" mkdir -p /mnt/ai-platform-data /mnt/feature-store /mnt/data
docker cp "$REPO_ROOT/data/." "$WORKER2:/mnt/ai-platform-data/"
docker cp "$REPO_ROOT/infra/feature-store/." "$WORKER2:/mnt/feature-store/"
docker cp "$REPO_ROOT/data/." "$WORKER2:/mnt/data/"
docker exec "$WORKER1" mkdir -p /mnt/data /mnt/docs
docker cp "$REPO_ROOT/data/." "$WORKER1:/mnt/data/"
docker cp "$REPO_ROOT/docs/." "$WORKER1:/mnt/docs/"

log "3/6 — building + importing images"
build_and_import() {
  local image="$1" dockerfile="$2"
  docker build -t "$image" -f "$dockerfile" "$REPO_ROOT"
  k3d image import "$image" -c "$CLUSTER"
}
build_and_import orchestration-api:openchoreo-local services/orchestration-api/Dockerfile
build_and_import ai-observability-server:openchoreo-local agents/mcp-servers/ai-observability-server/Dockerfile
build_and_import llmops-golden-paths-server:openchoreo-local agents/mcp-servers/llmops-golden-paths-server/Dockerfile
build_and_import golden-path-guide-server:openchoreo-local agents/mcp-servers/golden-path-guide-server/Dockerfile
build_and_import mlops-golden-paths-server:openchoreo-local agents/mcp-servers/mlops-golden-paths-server/Dockerfile
docker build -t feast-serve:openchoreo-local infra/feature-store/
k3d image import feast-serve:openchoreo-local -c "$CLUSTER"
# infra/argo-workflows/training-image/'s image — never pushed to a registry
# (see train-register-cluster-template.yaml) — re-import so it's present on
# worker2 too, not just whatever node it was originally imported to.
if docker image inspect training-image:local >/dev/null 2>&1; then
  k3d image import training-image:local -c "$CLUSTER"
fi

log "4/6 — AI Platform zone (worker2) — MLflow, Qdrant, MinIO, LiteLLM, Feast"
kubectl --context "$CTX" apply -f infra/ai-platform-zone/namespace.yaml
kubectl --context "$CTX" -n ai-platform-zone create configmap litellm-config \
  --from-file=config.yaml=infra/ai-platform-zone/litellm-config.yaml \
  --dry-run=client -o yaml | kubectl --context "$CTX" apply -f -
if [[ -f .env ]]; then set -a; source .env; set +a; fi
kubectl --context "$CTX" -n ai-platform-zone create secret generic litellm-secrets \
  --from-literal=anthropic-api-key="${ANTHROPIC_API_KEY:-}" \
  --from-literal=voyage-api-key="${VOYAGE_API_KEY:-}" \
  --from-literal=litellm-master-key="${LITELLM_MASTER_KEY:-sk-local-dev}" \
  --dry-run=client -o yaml | kubectl --context "$CTX" apply -f -
kubectl --context "$CTX" apply -f infra/ai-platform-zone/namespace.yaml -f infra/ai-platform-zone/mlflow.yaml \
  -f infra/ai-platform-zone/qdrant.yaml -f infra/ai-platform-zone/minio.yaml -f infra/ai-platform-zone/litellm.yaml \
  -f infra/ai-platform-zone/feast-serve.yaml
kubectl --context "$CTX" -n ai-platform-zone rollout status deployment/minio --timeout=120s
# --default-artifact-root s3://mlflow-artifacts (mlflow.yaml) needs this
# bucket to already exist — MinIO doesn't auto-create it. One-time-per-boot;
# harmless (mc mb) if it's already there.
kubectl --context "$CTX" -n ai-platform-zone exec deploy/minio -- \
  sh -c 'mc alias set local http://localhost:9000 minioadmin minioadmin >/dev/null && mc mb -p local/mlflow-artifacts'

log "5/6 — data plane portal (worker1) + workflow plane pin — OpenChoreo manifests"
# Cluster-scoped resources first (Kubernetes ClusterRole in infra/bootstrap/,
# OpenChoreo ClusterComponentType in platform-shared/), then the namespaced
# project resources that depend on them.
P=infra/openchoreo/namespaces/default/projects/platform
kubectl --context "$CTX" apply -f infra/bootstrap/clusterrole-orchestration-api.yaml
kubectl --context "$CTX" apply -f infra/openchoreo/platform-shared/component-types/clustercomponenttype-pinned-service.yaml
kubectl --context "$CTX" apply -f "$P/components/orchestration-api/networkpolicy-orchestration-api-workflows.yaml"
kubectl --context "$CTX" apply -f "$P/resource-mlflow.yaml"
kubectl --context "$CTX" apply -f "$P/resource-qdrant.yaml"
kubectl --context "$CTX" apply -f "$P/resource-litellm.yaml"
kubectl --context "$CTX" apply -f "$P/components/orchestration-api/component-orchestration-api.yaml"
kubectl --context "$CTX" apply -f "$P/components/orchestration-api/workload-orchestration-api.yaml"
for svc in ai-observability-server llmops-golden-paths-server golden-path-guide-server mlops-golden-paths-server; do
  kubectl --context "$CTX" apply -f "$P/components/${svc}/component-${svc}.yaml"
  kubectl --context "$CTX" apply -f "$P/components/${svc}/workload-${svc}.yaml"
done
# componentTypeEnvironmentConfigs (nodeSelector/tolerations/resources), not
# Component.spec.parameters, is what the renderer actually reads — see
# releasebinding-orchestration-api-development.yaml's comment. autoDeploy
# creates a bare ReleaseBinding on its own; these apply on top of it.
for svc in orchestration-api ai-observability-server llmops-golden-paths-server golden-path-guide-server mlops-golden-paths-server; do
  kubectl --context "$CTX" apply -f "$P/components/${svc}/releasebinding-${svc}-development.yaml"
done
kubectl --context "$CTX" apply -f infra/openchoreo/platform-shared/cluster-workflow-templates/argo/train-register-cluster-template.yaml

log "6/6 — retiring the old docker-compose stack"
if docker compose ls --format json 2>/dev/null | grep -q ai-delivery-portal; then
  docker compose -p ai-delivery-portal down
fi

log "done — verify with: kubectl --context $CTX get pods -A -o wide"
