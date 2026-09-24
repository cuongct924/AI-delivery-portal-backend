"""adapters/ai_platform/prediction_log_adapter.py — exercises SqlitePredictionLogAdapter
against a real sqlite file on tmp_path (no SDK to mock — same convention as
tests/test_qdrant_registry_adapter.py's QdrantVersionRegistryAdapter)."""

from adapters.ai_platform.prediction_log_adapter import SqlitePredictionLogAdapter


def _adapter(tmp_path) -> SqlitePredictionLogAdapter:
    return SqlitePredictionLogAdapter(path=str(tmp_path / "predictions.db"))


def test_list_predictions_returns_empty_list_for_unknown_model(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    assert adapter.list_predictions("fraud-detection") == []


def test_log_prediction_then_list_predictions_round_trips(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    adapter.log_prediction("fraud-detection", "3", {"amount": 100.0}, {"is_fraud": False})

    [entry] = adapter.list_predictions("fraud-detection")
    assert entry["model_name"] == "fraud-detection"
    assert entry["model_version"] == "3"
    assert entry["input"] == {"amount": 100.0}
    assert entry["output"] == {"is_fraud": False}
    assert entry["logged_at"]  # non-empty ISO 8601 timestamp


def test_log_prediction_tolerates_a_missing_output(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    adapter.log_prediction("fraud-detection", "3", {"amount": 100.0}, None)

    [entry] = adapter.list_predictions("fraud-detection")
    assert entry["output"] is None


def test_list_predictions_returns_newest_first(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    adapter.log_prediction("fraud-detection", "1", {"seq": 1}, None)
    adapter.log_prediction("fraud-detection", "2", {"seq": 2}, None)

    entries = adapter.list_predictions("fraud-detection")
    assert [e["input"]["seq"] for e in entries] == [2, 1]


def test_list_predictions_is_scoped_to_one_model_name(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    adapter.log_prediction("fraud-detection", "1", {}, None)
    adapter.log_prediction("churn-model", "1", {}, None)

    assert len(adapter.list_predictions("fraud-detection")) == 1
    assert len(adapter.list_predictions("churn-model")) == 1


def test_list_predictions_respects_limit(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    for i in range(5):
        adapter.log_prediction("fraud-detection", "1", {"seq": i}, None)

    assert len(adapter.list_predictions("fraud-detection", limit=2)) == 2


def test_state_persists_across_separate_adapter_instances(tmp_path) -> None:
    path = str(tmp_path / "predictions.db")
    SqlitePredictionLogAdapter(path=path).log_prediction("fraud-detection", "1", {"a": 1}, None)
    reloaded = SqlitePredictionLogAdapter(path=path)
    assert len(reloaded.list_predictions("fraud-detection")) == 1
