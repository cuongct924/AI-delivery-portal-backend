"""Rate tables used to turn a real consumed quantity into USD.

These are reference on-demand rates, not a live billing feed — every ledger
entry records the `unit_price` it used, so a rate change is auditable and a
real contract price can replace a constant here without touching callers.
"""

from typing import Final

# USD per GPU-hour, keyed by the same GPU types llm_serving/gpu_sizing.py
# sizes against. Reference on-demand list prices.
GPU_HOUR_USD: Final[dict[str, float]] = {
    "L4": 0.80,
    "L40S": 1.20,
    "A100": 3.20,
    "H100": 4.50,
    "H200": 6.00,
    "B200": 8.00,
}

# USD per CPU-hour for a training/monitoring job pod.
CPU_HOUR_USD: Final[float] = 0.05

# USD per 1k embedded chunks (voyage-3 reference rate).
EMBED_USD_PER_1K_CHUNK: Final[float] = 0.00012

# Nominal hours a training run takes, used to price a run at trigger time
# (the real duration is only known once the workflow finishes). Scaled by
# epochs and HPO trials so a bigger run costs more.
TRAIN_BASE_HOURS: Final[float] = 0.25

# Nominal hours a single monitoring cron run takes.
MONITOR_RUN_HOURS: Final[float] = 0.05

# USD per 1k judge/generation tokens (reference rate for a pre-flight estimate;
# the real per-call cost comes from LiteLLM's response header at run time).
TOKEN_USD_PER_1K: Final[float] = 0.0005

# Nominal hours a serving replica is priced for in a pre-flight estimate.
SERVE_ESTIMATE_HOURS: Final[float] = 24.0


def gpu_hour_price(gpu_type: str) -> float:
    return GPU_HOUR_USD.get(gpu_type, GPU_HOUR_USD["L4"])


def train_estimate_hours(epochs: int | None, num_trials: int | None) -> float:
    """Rough run length from the form's own cost drivers — epochs and HPO
    trials — so a 50-epoch, 20-trial search prices higher than a 5-epoch
    single run. Deliberately simple; the real duration replaces it later."""
    return TRAIN_BASE_HOURS * max(1, epochs or 1) * max(1, num_trials or 1)


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def estimate_golden_path_cost(
    golden_path: str, stage: str, params: dict[str, object]
) -> tuple[float, dict[str, float]]:
    """Pre-flight estimate for a golden-path run, from the same rate tables the
    real ledger uses. Deliberately simple — the run's own recorded event
    replaces it once it executes. Returns (total_usd, per-component breakdown).
    """
    breakdown: dict[str, float] = {}
    match golden_path:
        case "train-track-register":
            hours = train_estimate_hours(
                _as_int(params.get("epochs"), 1), _as_int(params.get("numTrials"), 1)
            )
            gpu_type = str(params.get("gpuType") or "")
            if gpu_type and gpu_type != "none":
                breakdown["gpu"] = hours * gpu_hour_price(gpu_type)
            else:
                breakdown["cpu"] = hours * CPU_HOUR_USD
        case "llm-draft-ingest":
            if str(params.get("artifactKind") or "") == "rag-index":
                chunks = _as_int(params.get("chunkCount"), 1000)
                breakdown["embedding"] = chunks / 1000 * EMBED_USD_PER_1K_CHUNK
            else:
                tokens = _as_int(params.get("tokens"), 2000)
                breakdown["token"] = tokens / 1000 * TOKEN_USD_PER_1K
        case "llm-evaluate-activate":
            tokens = _as_int(params.get("tokens"), 5000)
            breakdown["token"] = tokens / 1000 * TOKEN_USD_PER_1K
        case "evaluate-deploy-model":
            breakdown["cpu"] = SERVE_ESTIMATE_HOURS * CPU_HOUR_USD
        case "llm-serve-deploy":
            gpu_type = str(params.get("gpuType") or "L4")
            gpu_count = _as_int(params.get("gpuCount"), 1)
            breakdown["gpu"] = gpu_count * SERVE_ESTIMATE_HOURS * gpu_hour_price(gpu_type)
        case "setup-model-monitoring":
            runs = _as_int(params.get("runsPerMonth"), 30)
            breakdown["cpu"] = runs * MONITOR_RUN_HOURS * CPU_HOUR_USD
        case _:
            breakdown["cpu"] = CPU_HOUR_USD

    total = round(sum(breakdown.values()), 4)
    return total, {key: round(value, 4) for key, value in breakdown.items()}
