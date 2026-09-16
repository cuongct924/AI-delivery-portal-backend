#!/usr/bin/env bash
# Local stand-in for a GitOps controller: polls this repo's `main` branch on
# GitHub, and the moment a commit lands there that changes
# infra/openchoreo/telco-fraud-detection/workload-serving.yaml, applies
# that exact file to the cluster with `kubectl apply`.
#
# Why a script instead of a GitHub Actions workflow: GitHub's runners
# (cloud) have no network path to this machine's local k3d cluster — the
# API server is only reachable from this machine (https://0.0.0.0:6550
# directly, or https://host.docker.internal:6550 from inside a container
# on this same Docker Desktop). A cloud CI workflow could never reach it,
# and a self-hosted runner is more setup than this repo needs for a local
# demo. Since everything already runs on one machine, polling from that
# same machine is the simplest thing that actually works.
#
# "Gated" still means something real: this only ever applies what's
# already on `main` — the actual approval is merging the PR on GitHub, the
# same gate a real GitOps controller would honor. It never applies a
# feature branch, and never touches the file's *content* on its own —
# apply-exactly-what's-in-git is the whole job.
#
# Usage:
#   ./scripts/watch-and-apply-workload.sh            # poll forever (Ctrl-C to stop)
#   ./scripts/watch-and-apply-workload.sh --once      # check once and exit (cron-friendly)
#   ./scripts/watch-and-apply-workload.sh --interval 60
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WATCH_PATH="infra/openchoreo/telco-fraud-detection/workload-serving.yaml"
STATE_DIR="$REPO_ROOT/scripts/.watch-state"
STATE_FILE="$STATE_DIR/last-applied-sha"
INTERVAL=30
ONCE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once) ONCE=true; shift ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    *) echo "Usage: $0 [--once] [--interval SECONDS]" >&2; exit 1 ;;
  esac
done

mkdir -p "$STATE_DIR"
touch "$STATE_FILE"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

check_and_apply() {
  if ! git -C "$REPO_ROOT" fetch origin main --quiet; then
    log "WARNING: git fetch origin main failed — skipping this poll"
    return 0
  fi

  local remote_sha last_applied
  remote_sha="$(git -C "$REPO_ROOT" rev-parse origin/main)"
  last_applied="$(cat "$STATE_FILE" 2>/dev/null || true)"

  if [[ "$remote_sha" == "$last_applied" ]]; then
    return 0
  fi

  # Only the FIRST poll (empty state file) has no prior SHA to diff
  # against — every later poll skips the apply entirely when this exact
  # path didn't change, so an unrelated commit landing on main is a no-op.
  if [[ -n "$last_applied" ]] \
    && git -C "$REPO_ROOT" diff --quiet "$last_applied" "$remote_sha" -- "$WATCH_PATH" 2>/dev/null; then
    log "origin/main moved to $remote_sha, but $WATCH_PATH is unchanged — nothing to apply"
    echo "$remote_sha" > "$STATE_FILE"
    return 0
  fi

  local content
  if ! content="$(git -C "$REPO_ROOT" show "$remote_sha:$WATCH_PATH" 2>/dev/null)"; then
    log "WARNING: $WATCH_PATH doesn't exist at $remote_sha (deleted?) — nothing to apply"
    echo "$remote_sha" > "$STATE_FILE"
    return 0
  fi

  # Cheap sanity check — the PR review/merge is the real approval, this is
  # just a guard against applying whatever happens to land at this exact
  # path if it's ever not actually a Workload manifest.
  if ! grep -q '^kind: Workload$' <<< "$content"; then
    log "REFUSING to apply $remote_sha:$WATCH_PATH — no 'kind: Workload' line, doesn't look like a Workload manifest"
    return 1
  fi

  log "Applying $WATCH_PATH from $remote_sha"
  if kubectl apply -f - <<< "$content"; then
    echo "$remote_sha" > "$STATE_FILE"
    log "Applied $remote_sha"
  else
    log "kubectl apply failed for $remote_sha — will retry next poll (state not advanced)"
    return 1
  fi
}

log "Watching origin/main for changes to $WATCH_PATH"
if $ONCE; then
  check_and_apply
else
  log "Polling every ${INTERVAL}s — Ctrl-C to stop"
  while true; do
    check_and_apply || log "poll failed — retrying in ${INTERVAL}s"
    sleep "$INTERVAL"
  done
fi
