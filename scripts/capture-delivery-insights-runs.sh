#!/usr/bin/env bash
# Captures real Golden Path workflow runs from the local k3d cluster into
# services/orchestration-api/mock_data/delivery_insights_runs.json, which
# routers/delivery_insights.py serves to the Portal's Delivery Insights page.
#
# Re-run after triggering Golden Paths (train-track-register / evaluate-deploy-model
# / setup-model-monitoring) so the dashboard reflects the latest real runs.
#
# Usage: bash scripts/capture-delivery-insights-runs.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO_ROOT/services/orchestration-api/mock_data/delivery_insights_runs.json"
NAMESPACE="${NAMESPACE:-default}"

mkdir -p "$(dirname "$OUT")"

kubectl get workflowruns -n "$NAMESPACE" -o json | python3 -c '
import json, sys
from datetime import datetime, UTC

namespace = sys.argv[1]
doc = json.load(sys.stdin)
runs = []
for item in doc.get("items", []):
    status = item.get("status", {})
    conditions = {c.get("type"): c.get("status") for c in status.get("conditions", [])}
    if conditions.get("WorkflowSucceeded") == "True":
        outcome = "success"
    elif conditions.get("WorkflowFailed") == "True":
        outcome = "failure"
    else:
        continue
    steps = [
        {
            "name": t.get("name"),
            "started_at": t.get("startedAt"),
            "finished_at": t.get("completedAt"),
        }
        for t in status.get("tasks", [])
        if t.get("startedAt") and t.get("completedAt")
    ]
    runs.append(
        {
            "name": item["metadata"]["name"],
            "started_at": status.get("startedAt"),
            "finished_at": status.get("completedAt"),
            "outcome": outcome,
            "steps": steps,
        }
    )

runs.sort(key=lambda r: r["started_at"] or "")
json.dump(
    {
        "captured_at": datetime.now(UTC).isoformat(),
        "source": f"k3d openchoreo-quick-start / namespace {namespace}",
        "runs": runs,
    },
    sys.stdout,
    indent=2,
)
' "$NAMESPACE" > "$OUT"

python3 - "$OUT" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
runs = doc["runs"]
ok = sum(1 for r in runs if r["outcome"] == "success")
print(f"captured {len(runs)} runs ({ok} success, {len(runs) - ok} failure) -> {sys.argv[1]}")
PY