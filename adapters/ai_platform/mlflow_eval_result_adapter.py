"""MLflow-backed implementation of IEvalResultAdapter — persists LLM-as-a-judge
evaluation results as MLflow runs, following the same "monitoring run" pattern
as infra/argo-workflows/training-image/monitor_drift.py (lines 98-107).
"""

from datetime import datetime

import mlflow
import pandas as pd

from adapters.ai_platform._mlflow import get_mlflow_tracking_uri
from adapters.ai_platform.interfaces import IEvalResultAdapter


class MlflowEvalResultAdapter(IEvalResultAdapter):
    def __init__(self) -> None:
        mlflow.set_tracking_uri(get_mlflow_tracking_uri())

    def log_judge_result(
        self,
        kind: str,
        name: str,
        version: str,
        judge_result: object,
        passed: bool,
    ) -> None:
        run_name = f"judge-{kind}-{name}-v{version}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("eval_kind", kind)
            mlflow.set_tag("eval_name", name)
            mlflow.set_tag("eval_version", version)
            mlflow.set_tag("passed", str(passed))

            jr = judge_result if isinstance(judge_result, dict) else {}
            safety = jr.get("safety")
            correctness = jr.get("correctness")
            relevance = jr.get("relevance")
            if safety is not None:
                mlflow.log_metric("safety", float(safety))  # type: ignore[arg-type]
            if correctness is not None:
                mlflow.log_metric("correctness", float(correctness))  # type: ignore[arg-type]
            if relevance is not None:
                mlflow.log_metric("relevance", float(relevance))  # type: ignore[arg-type]

    def get_last_failure_at(self, kind: str, name: str) -> datetime | None:
        filter_string = (
            f"tags.eval_kind = '{kind}' and tags.eval_name = '{name}' and tags.passed = 'False'"
        )
        runs = mlflow.search_runs(
            filter_string=filter_string,
            order_by=["start_time DESC"],
            max_results=1,
        )
        if isinstance(runs, list):
            runs = pd.DataFrame(runs)
        if runs.empty:
            return None
        start_time = runs.iloc[0]["start_time"]  # type: ignore[attr-defined]
        if isinstance(start_time, str):
            return datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        return start_time
