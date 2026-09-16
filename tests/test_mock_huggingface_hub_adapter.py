"""adapters/mock_huggingface_hub_adapter.py."""

from adapters.mock_huggingface_hub_adapter import MockHuggingFaceHubAdapter


def test_known_gated_model_returns_real_architecture() -> None:
    info = MockHuggingFaceHubAdapter().get_model_info("meta-llama/Llama-3.1-8B-Instruct")
    assert info["exists"] is True
    assert info["is_gated"] is True
    assert info["num_layers"] == 32


def test_known_not_found_model_returns_exists_false() -> None:
    info = MockHuggingFaceHubAdapter().get_model_info("does-not-exist/not-a-real-model")
    assert info["exists"] is False


def test_unknown_model_id_returns_synthetic_ungated_info() -> None:
    info = MockHuggingFaceHubAdapter().get_model_info("some-org/some-model")
    assert info["exists"] is True
    assert info["is_gated"] is False
    assert info["param_count_billion"] == 7.0
