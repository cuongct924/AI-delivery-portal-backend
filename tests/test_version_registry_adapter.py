"""adapters/ai_platform/version_registry_adapter.py — exercises JsonFileVersionRegistryAdapter
against a real file on tmp_path (no SDK to mock — same convention as other
pure-I/O adapters with no existing mock precedent)."""

import pytest

from adapters.ai_platform.version_registry_adapter import JsonFileVersionRegistryAdapter


def _adapter(tmp_path) -> JsonFileVersionRegistryAdapter:
    return JsonFileVersionRegistryAdapter(path=str(tmp_path / "registry.json"))


def test_register_version_starts_at_1_and_increments(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    v1 = adapter.register_version("prompt", "mlops", {"content": "v1"})
    v2 = adapter.register_version("prompt", "mlops", {"content": "v2"})
    assert v1 == "1"
    assert v2 == "2"


def test_register_version_is_independent_per_kind_and_name(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    adapter.register_version("prompt", "mlops", {"content": "a"})
    first_rag_version = adapter.register_version("rag-index", "mlops", {"chunks_ingested": 1})
    assert first_rag_version == "1"  # separate counter — same name, different kind


def test_get_active_version_returns_none_before_any_activation(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    adapter.register_version("prompt", "mlops", {"content": "v1"})
    assert adapter.get_active_version("prompt", "mlops") is None


def test_set_active_version_then_get_active_version_round_trips(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    version = adapter.register_version("prompt", "mlops", {"content": "v1"})
    adapter.set_active_version("prompt", "mlops", version)
    assert adapter.get_active_version("prompt", "mlops") == version


def test_set_active_version_raises_for_unregistered_name(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    with pytest.raises(ValueError, match="no registered versions"):
        adapter.set_active_version("prompt", "unknown", "1")


def test_set_active_version_raises_for_unregistered_version(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    adapter.register_version("prompt", "mlops", {"content": "v1"})
    with pytest.raises(ValueError, match="no version '2'"):
        adapter.set_active_version("prompt", "mlops", "2")


def test_get_version_raises_for_unknown_version(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    adapter.register_version("prompt", "mlops", {"content": "v1"})
    with pytest.raises(ValueError, match="no version '9'"):
        adapter.get_version("prompt", "mlops", "9")


def test_list_versions_returns_all_registered_versions(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    adapter.register_version("prompt", "mlops", {"content": "v1"})
    adapter.register_version("prompt", "mlops", {"content": "v2"})
    versions = adapter.list_versions("prompt", "mlops")
    assert set(versions) == {"1", "2"}


def test_list_names_returns_every_registered_name_for_a_kind(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    adapter.register_version("prompt", "mlops", {"content": "a"})
    adapter.register_version("prompt", "k8s", {"content": "b"})
    assert set(adapter.list_names("prompt")) == {"mlops", "k8s"}


def test_list_names_returns_empty_list_for_unknown_kind(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    assert adapter.list_names("rag-index") == []


def test_state_persists_across_separate_adapter_instances(tmp_path) -> None:
    path = str(tmp_path / "registry.json")
    JsonFileVersionRegistryAdapter(path=path).register_version("prompt", "mlops", {"content": "v1"})
    reloaded = JsonFileVersionRegistryAdapter(path=path)
    assert reloaded.list_versions("prompt", "mlops") == {"1": {"content": "v1"}}
