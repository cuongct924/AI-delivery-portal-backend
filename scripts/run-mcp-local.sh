#!/usr/bin/env bash
# Quickly run a single MCP server locally to test via the terminal (streamable-http transport).
# Usage: bash scripts/run-mcp-local.sh ai-observability   (or: llmops-golden-paths|golden-path-guide|mlops-golden-paths)
set -e

SERVER=$1
if [ -z "$SERVER" ]; then
  echo "Usage: bash scripts/run-mcp-local.sh <ai-observability|llmops-golden-paths|golden-path-guide|mlops-golden-paths>"
  exit 1
fi

case "$SERVER" in
  ai-observability)   DIR="agents/mcp-servers/ai-observability-server" ;;
  llmops-golden-paths) DIR="agents/mcp-servers/llmops-golden-paths-server" ;;
  golden-path-guide) DIR="agents/mcp-servers/golden-path-guide-server" ;;
  mlops-golden-paths) DIR="agents/mcp-servers/mlops-golden-paths-server" ;;
  *) echo "Unrecognized: $SERVER (only ai-observability|llmops-golden-paths|golden-path-guide|mlops-golden-paths are supported)"; exit 1 ;;
esac

VENV_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "$VENV_PY not found — run 'make install' first" >&2
  exit 1
fi

echo "=== Installing dependencies for $SERVER-server ==="
"$VENV_PY" -m pip install -q -r "$DIR/requirements.txt"

echo "=== Running $SERVER-server (PYTHONPATH=. so adapters/ can be imported) ==="
PYTHONPATH=. "$VENV_PY" "$DIR/server.py"
