"""The one helper every golden-path step calls to record a cost event.

Kept deliberately small and side-effect-only: a step that spends money calls
`record_cost_event(...)` and moves on. Recording never raises — a ledger write
failure must not fail the training run / eval / deploy that produced it.
"""

import logging
from datetime import UTC, datetime

from adapters.ai_platform.interfaces import CostLedgerEntry
from adapters.factory import get_cost_adapter

logger = logging.getLogger(__name__)

# The catalog namespace OpenChoreo entities live in; the observer cost API is
# queried per namespace, so entries carry it to be matched.
DEFAULT_NAMESPACE = "default"


def record_cost_event(
    *,
    stage: str,
    artifact_kind: str,
    artifact_id: str,
    cost_usd: float,
    source: str,
    version: str = "",
    environment: str = "",
    team: str = "",
    business_domain: str = "",
    quantity: float = 0.0,
    unit: str = "",
    unit_price: float = 0.0,
    run_id: str = "",
    namespace: str = DEFAULT_NAMESPACE,
    timestamp: str | None = None,
) -> None:
    """Append one attributed cost event. `stage` is "build" | "gate" | "run"."""
    entry = CostLedgerEntry(
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        stage=stage,
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        version=version,
        environment=environment,
        namespace=namespace,
        team=team,
        business_domain=business_domain,
        quantity=quantity,
        unit=unit,
        unit_price=unit_price,
        cost_usd=cost_usd,
        source=source,
        run_id=run_id,
    )
    try:
        get_cost_adapter().record_cost(entry)
    except Exception as exc:  # noqa: BLE001 — never fail the caller
        logger.warning("Failed to record cost event %s/%s: %s", stage, artifact_id, exc)
