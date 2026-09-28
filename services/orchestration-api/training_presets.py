"""Known-use-case defaults for train-track-register's `useCase` field —
mirrors AI-delivery-portal-frontend's
`packages/app/src/modules/scaffolder/trainingPresets.ts` (the Scaffolder
form's own auto-fill for the same field) so the chat agent doesn't ask
about, or guess differently than, what the form would silently fill in
for the same use case. A deliberate second copy, not a shared import —
the two repos don't share a Python/TS boundary; keep them in sync by
hand if the frontend file changes (small, stable, editorial data, not
business logic that's likely to drift often).
"""

from typing import Final, TypedDict


class TrainingPreset(TypedDict):
    modelName: str
    algorithm: str
    targetColumn: str | None
    idColumns: list[str]
    timeColumn: str | None


TRAINING_PRESETS: Final[dict[str, TrainingPreset]] = {
    "telco-fraud-detection": {
        "modelName": "telco-fraud-detection",
        "algorithm": "RandomForestClassifier",
        "targetColumn": "is_fraud",
        "idColumns": ["transaction_id"],
        "timeColumn": None,
    },
    "network-anomaly-detection": {
        "modelName": "network-anomaly-detection",
        "algorithm": "IsolationForest",
        "targetColumn": "is_anomaly",
        "idColumns": ["reading_id"],
        "timeColumn": None,
    },
    "customer-segmentation": {
        "modelName": "customer-segmentation",
        "algorithm": "KMeans",
        "targetColumn": None,
        "idColumns": ["customer_id"],
        "timeColumn": None,
    },
    "house-price-prediction": {
        "modelName": "house-price-prediction",
        "algorithm": "RandomForestRegressor",
        "targetColumn": "price",
        "idColumns": [],
        "timeColumn": None,
    },
    "revenue-forecast": {
        "modelName": "revenue-forecast",
        "algorithm": "LinearRegression",
        "targetColumn": "revenue",
        "idColumns": [],
        "timeColumn": "date",
    },
}


def render_training_presets() -> str:
    """Condensed, prompt-embeddable text form of TRAINING_PRESETS — one
    line per use case, only the non-null fields."""
    lines: list[str] = []
    for use_case, preset in TRAINING_PRESETS.items():
        parts = [f"algorithm={preset['algorithm']}"]
        if preset["targetColumn"] is not None:
            parts.append(f"targetColumn={preset['targetColumn']}")
        if preset["idColumns"]:
            parts.append(f"idColumns={preset['idColumns']}")
        if preset["timeColumn"] is not None:
            parts.append(f"timeColumn={preset['timeColumn']}")
        lines.append(f"- {use_case}: {', '.join(parts)}")
    return "\n".join(lines)
