#!/usr/bin/env bash
# Quickly run a single MCP server locally to test via the terminal (streamable-http transport).
# Usage: bash scripts/run-mcp-local.sh observability   (or: golden-paths|golden-path-guide)
set -e

SERVER=$1
if [ -z "$SERVER" ]; then
  echo "Usage: bash scripts/run-mcp-local.sh <observability|golden-paths|golden-path-guide>"
  exit 1
fi

case "$SERVER" in
  observability)   DIR="agents/mcp-servers/observability-server" ;;
  golden-paths)     DIR="agents/mcp-servers/golden-paths-server" ;;
  golden-path-guide) DIR="agents/mcp-servers/golden-path-guide-server" ;;
  *) echo "Unrecognized: $SERVER (only observability|golden-paths|golden-path-guide are supported)"; exit 1 ;;
esac

echo "=== Installing dependencies for $SERVER-server ==="
pip install -q -r "$DIR/requirements.txt"

echo "=== Running $SERVER-server (PYTHONPATH=. so adapters/ can be imported) ==="
PYTHONPATH=. python "$DIR/server.py"
