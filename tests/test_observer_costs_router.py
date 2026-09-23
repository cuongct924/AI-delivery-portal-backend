"""services/orchestration-api/routers/observer_costs.py — the mock OpenChoreo
observer cost contract the Cost Insights page calls. Pure synthetic data, so
the functions are called directly (no adapter to mock)."""

from routers.observer_costs import get_cost_recommendations, get_costs


def test_get_costs_returns_items_for_the_namespace_scope() -> None:
    response = get_costs(namespace="default", environment="development")
    assert len(response.items) > 0
    assert all(item.namespace == "default" for item in response.items)
    assert all(item.environment == "development" for item in response.items)


def test_get_costs_narrows_to_a_single_component() -> None:
    response = get_costs(
        namespace="default",
        environment="development",
        project="telco-fraud-detection",
        component="serving",
    )
    assert len(response.items) == 1
    assert response.items[0].project == "telco-fraud-detection"
    assert response.items[0].component == "serving"


def test_get_costs_is_deterministic() -> None:
    first = get_costs(namespace="default", environment="development")
    second = get_costs(namespace="default", environment="development")
    assert [i.cpuCost for i in first.items] == [i.cpuCost for i in second.items]


def test_get_costs_buckets_when_granularity_is_given() -> None:
    response = get_costs(
        namespace="default",
        environment="development",
        startTime="2026-07-01T00:00:00.000Z",
        endTime="2026-07-01T06:00:00.000Z",
        granularity="1h",
    )
    # 6 one-hour buckets per target, so more items than the un-bucketed call.
    assert len(response.items) > 6
    assert all(item.cpuCost >= 0 for item in response.items)


def test_get_cost_recommendations_offer_a_lower_right_size() -> None:
    response = get_cost_recommendations(
        namespace="default",
        environment="development",
        project="telco-fraud-detection",
        component="serving",
    )
    assert len(response.items) == 1
    item = response.items[0]
    assert item.recommendation.cpuCost < item.current.cpuCost
    assert item.recommendation.memoryCost < item.current.memoryCost
