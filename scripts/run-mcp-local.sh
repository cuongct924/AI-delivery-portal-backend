#!/usr/bin/env bash
# Quickly run a single MCP server locally to test via the terminal (streamable-http transport).
# Usage: bash scripts/run-mcp-local.sh observability   (or: llmops-golden-paths|golden-path-guide|mlops-golden-paths)
set -e

SERVER=$1
if [ -z "$SERVER" ]; then
  echo "Usage: bash scripts/run-mcp-local.sh <observability|llmops-golden-paths|golden-path-guide|mlops-golden-paths>"
  exit 1
fi

case "$SERVER" in
  observability)   DIR="agents/mcp-servers/observability-server" ;;
  llmops-golden-paths) DIR="agents/mcp-servers/llmops-golden-paths-server" ;;
  golden-path-guide) DIR="agents/mcp-servers/golden-path-guide-server" ;;
  mlops-golden-paths) DIR="agents/mcp-servers/mlops-golden-paths-server" ;;
  *) echo "Unrecognized: $SERVER (only observability|llmops-golden-paths|golden-path-guide|mlops-golden-paths are supported)"; exit 1 ;;
esac

echo "=== Installing dependencies for $SERVER-server ==="
pip install -q -r "$DIR/requirements.txt"

echo "=== Running $SERVER-server (PYTHONPATH=. so adapters/ can be imported) ==="
PYTHONPATH=. python "$DIR/server.py"
