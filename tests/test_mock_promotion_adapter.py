"""Tests adapters/delivery/mock_promotion_adapter.py."""

import pytest

from adapters.delivery.mock_promotion_adapter import MockPromotionAdapter


@pytest.fixture
def adapter() -> MockPromotionAdapter:
    return MockPromotionAdapter()


def test_seeds_development_only(adapter: MockPromotionAdapter) -> None:
    status = adapter.get_promotion_status()

    assert status["environments"] == {"development": "1", "staging": None, "production": None}
    assert status["prod_pending_approval"] is False


def test_promote_to_staging_copies_the_development_release(adapter: MockPromotionAdapter) -> None:
    status = adapter.promote("staging")

    assert status["environments"]["staging"] == "1"
    assert status["prod_pending_approval"] is True  # staging ahead of (unset) production


def test_promote_to_production_requires_staging_first(adapter: MockPromotionAdapter) -> None:
    with pytest.raises(ValueError, match="no release bound in 'staging'"):
        adapter.promote("production")


def test_promote_to_production_after_staging_clears_pending_approval(
    adapter: MockPromotionAdapter,
) -> None:
    adapter.promote("staging")

    status = adapter.promote("production")

    assert status["environments"]["production"] == "1"
    assert status["prod_pending_approval"] is False


def test_promote_rejects_an_unknown_target_environment(adapter: MockPromotionAdapter) -> None:
    with pytest.raises(ValueError, match="isn't a valid promotion target"):
        adapter.promote("development")


def test_rollback_promotion_undoes_the_last_promote(adapter: MockPromotionAdapter) -> None:
    adapter.promote("staging")  # staging: None -> "1"
    adapter._bindings["development"] = "2"
    adapter.promote("staging")  # staging: "1" -> "2", previous recorded as "1"

    status = adapter.rollback_promotion("staging")

    assert status["environments"]["staging"] == "1"


def test_rollback_promotion_raises_when_nothing_bound_yet(adapter: MockPromotionAdapter) -> None:
    with pytest.raises(ValueError, match="no release bound"):
        adapter.rollback_promotion("staging")


def test_rollback_promotion_raises_when_nothing_was_ever_promoted_over(
    adapter: MockPromotionAdapter,
) -> None:
    adapter.promote("staging")  # first-ever binding — nothing to roll back to

    with pytest.raises(ValueError, match="no prior release recorded"):
        adapter.rollback_promotion("staging")


def test_rollback_promotion_is_itself_reversible(adapter: MockPromotionAdapter) -> None:
    adapter.promote("staging")
    adapter._bindings["development"] = "2"
    adapter.promote("staging")

    adapter.rollback_promotion("staging")  # staging: "2" -> "1"
    status = adapter.rollback_promotion("staging")  # staging: "1" -> "2"

    assert status["environments"]["staging"] == "2"
