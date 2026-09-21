"""Compares metrics between two model versions to detect drift.

Reusable from both MCP tools (agents/mcp-servers/) and regular FastAPI routes
(services/orchestration-api/) — the business logic is written once, here.
"""

from adapters.ai_platform.mlflow_adapter import MlflowAdapter


def evaluate_drift(model_name: str, current_version: str, baseline_version: str) -> dict:
    adapter = MlflowAdapter()
    current = adapter.get_model_metrics(model_name, current_version)
    baseline = adapter.get_model_metrics(model_name, baseline_version)

    drift = {}
    for metric_name, baseline_value in baseline.items():
        current_value = current.get(metric_name)
        if current_value is None:
            continue
        delta_pct = (
            (current_value - baseline_value) / baseline_value * 100 if baseline_value else None
        )
        drift[metric_name] = {
            "baseline": baseline_value,
            "current": current_value,
            "delta_pct": delta_pct,
        }
    return drift
