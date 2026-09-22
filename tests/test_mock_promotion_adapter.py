"""Tests adapters/delivery/mock_promotion_adapter.py."""

import pytest

from adapters.delivery.interfaces import PromotionStatus
from adapters.delivery.mock_promotion_adapter import MockPromotionAdapter


@pytest.fixture
def adapter() -> MockPromotionAdapter:
    return MockPromotionAdapter()


def _promote(adapter: MockPromotionAdapter, target_environment: str) -> PromotionStatus:
    """Test-only helper mirroring what routers/models.py does across 2
    requests (resolve, then confirm) — real callers always split these."""
    release = adapter.resolve_promotion_release(target_environment)
    return adapter.confirm_promotion(target_environment, release)


def _rollback(adapter: MockPromotionAdapter, environment: str) -> PromotionStatus:
    release = adapter.resolve_rollback_release(environment)
    return adapter.confirm_promotion(environment, release)


def test_seeds_development_only(adapter: MockPromotionAdapter) -> None:
    status = adapter.get_promotion_status()

    assert status["environments"] == {"development": "1", "staging": None, "production": None}
    assert status["prod_pending_approval"] is False


def test_promote_to_staging_copies_the_development_release(adapter: MockPromotionAdapter) -> None:
    status = _promote(adapter, "staging")

    assert status["environments"]["staging"] == "1"
    assert status["prod_pending_approval"] is True  # staging ahead of (unset) production


def test_promote_to_production_requires_staging_first(adapter: MockPromotionAdapter) -> None:
    with pytest.raises(ValueError, match="no release bound in 'staging'"):
        adapter.resolve_promotion_release("production")


def test_promote_to_production_after_staging_clears_pending_approval(
    adapter: MockPromotionAdapter,
) -> None:
    _promote(adapter, "staging")

    status = _promote(adapter, "production")

    assert status["environments"]["production"] == "1"
    assert status["prod_pending_approval"] is False


def test_promote_rejects_an_unknown_target_environment(adapter: MockPromotionAdapter) -> None:
    with pytest.raises(ValueError, match="isn't a valid promotion target"):
        adapter.resolve_promotion_release("development")


def test_resolve_promotion_release_does_not_write_anything(adapter: MockPromotionAdapter) -> None:
    adapter.resolve_promotion_release("staging")

    assert adapter.get_promotion_status()["environments"]["staging"] is None


def test_rollback_promotion_undoes_the_last_promote(adapter: MockPromotionAdapter) -> None:
    _promote(adapter, "staging")  # staging: None -> "1"
    adapter._bindings["development"] = "2"
    _promote(adapter, "staging")  # staging: "1" -> "2", previous recorded as "1"

    status = _rollback(adapter, "staging")

    assert status["environments"]["staging"] == "1"


def test_rollback_promotion_raises_when_nothing_bound_yet(adapter: MockPromotionAdapter) -> None:
    with pytest.raises(ValueError, match="no release bound"):
        adapter.resolve_rollback_release("staging")


def test_rollback_promotion_raises_when_nothing_was_ever_promoted_over(
    adapter: MockPromotionAdapter,
) -> None:
    _promote(adapter, "staging")  # first-ever binding — nothing to roll back to

    with pytest.raises(ValueError, match="no prior release recorded"):
        adapter.resolve_rollback_release("staging")


def test_rollback_promotion_is_itself_reversible(adapter: MockPromotionAdapter) -> None:
    _promote(adapter, "staging")
    adapter._bindings["development"] = "2"
    _promote(adapter, "staging")

    _rollback(adapter, "staging")  # staging: "2" -> "1"
    status = _rollback(adapter, "staging")  # staging: "1" -> "2"

    assert status["environments"]["staging"] == "2"
