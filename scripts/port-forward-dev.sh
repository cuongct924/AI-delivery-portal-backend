#!/usr/bin/env bash
# Single entry point for everything a host-run orchestration-api
# (`make run-orchestration-api`) needs reachable: the AI Platform zone
# (worker2 — Qdrant/MLflow/LiteLLM/MinIO/Feast) and the platform project's
# MCP servers (worker1). Both are self-healing (see each script's own
# supervise_forward) — a dropped forward reconnects on its own instead of
# silently staying dead until someone notices.
#
# Run this in its own terminal and leave it up; Ctrl-C tears everything
# down, including each sub-script's own forwards.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

pids=()
cleanup() {
  for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

bash scripts/port-forward-ai-platform.sh &
pids+=("$!")
bash scripts/port-forward-mcp-servers.sh &
pids+=("$!")

echo "=== all dev port-forwards up, self-healing. Ctrl-C to stop everything. ==="
wait
