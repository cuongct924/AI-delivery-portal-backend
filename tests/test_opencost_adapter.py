"""adapters/ai_platform/opencost_adapter.py — mocked httpx, no real OpenCost."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from adapters.ai_platform.opencost_adapter import fetch_allocation, to_cost_items

_NOW = datetime.now(UTC)


def test_fetch_allocation_returns_none_without_url(monkeypatch) -> None:
    monkeypatch.delenv("OPENCOST_URL", raising=False)
    assert fetch_allocation("default", _NOW, _NOW) is None


def test_fetch_allocation_maps_the_response(monkeypatch) -> None:
    monkeypatch.setenv("OPENCOST_URL", "http://opencost.test")
    response = MagicMock()
    response.json.return_value = {
        "data": [
            {
                "default/serving": {
                    "cpuCost": 1.5,
                    "ramCost": 0.5,
                    "efficiency": 0.4,
                    "properties": {"namespace": "default", "controller": "serving"},
                    "window": {"start": "a", "end": "b"},
                }
            }
        ]
    }
    with patch("adapters.ai_platform.opencost_adapter.httpx.get", return_value=response):
        data = fetch_allocation("default", _NOW, _NOW)

    assert data is not None
    items = to_cost_items(data, "development")
    assert items[0]["component"] == "serving"
    assert items[0]["cpuCost"] == 1.5
    assert items[0]["memoryCost"] == 0.5
    assert items[0]["environment"] == "development"


def test_fetch_allocation_falls_back_on_error(monkeypatch) -> None:
    monkeypatch.setenv("OPENCOST_URL", "http://opencost.test")
    with patch(
        "adapters.ai_platform.opencost_adapter.httpx.get",
        side_effect=Exception("boom"),
    ):
        assert fetch_allocation("default", _NOW, _NOW) is None
