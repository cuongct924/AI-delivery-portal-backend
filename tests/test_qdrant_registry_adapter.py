"""adapters/ai_platform/qdrant_registry_adapter.py — exercises
QdrantVersionRegistryAdapter against a real Qdrant (QDRANT_URL, default
http://localhost:6333), same "no SDK to mock" convention as the JSON adapter
it replaced. Each test gets its own collection so runs don't pollute each
other (Qdrant persists across runs, unlike the old tmp_path file)."""

import uuid
from collections.abc import Iterator

import pytest

from adapters.ai_platform.qdrant_registry_adapter import QdrantVersionRegistryAdapter


@pytest.fixture
def adapter() -> Iterator[QdrantVersionRegistryAdapter]:
    collection = f"_llmops_registry_test_{uuid.uuid4().hex[:8]}"
    instance = QdrantVersionRegistryAdapter(collection=collection)
    yield instance
    instance.client.delete_collection(collection)


def test_register_version_starts_at_1_and_increments(adapter) -> None:
    v1 = adapter.register_version("prompt", "mlops", {"content": "v1"})
    v2 = adapter.register_version("prompt", "mlops", {"content": "v2"})
    assert v1 == "1"
    assert v2 == "2"


def test_register_version_is_independent_per_kind_and_name(adapter) -> None:
    adapter.register_version("prompt", "mlops", {"content": "a"})
    first_rag_version = adapter.register_version("rag-index", "mlops", {"chunks_ingested": 1})
    assert first_rag_version == "1"  # separate counter — same name, different kind


def test_get_active_version_returns_none_before_any_activation(adapter) -> None:
    adapter.register_version("prompt", "mlops", {"content": "v1"})
    assert adapter.get_active_version("prompt", "mlops") is None


def test_set_active_version_then_get_active_version_round_trips(adapter) -> None:
    version = adapter.register_version("prompt", "mlops", {"content": "v1"})
    adapter.set_active_version("prompt", "mlops", version)
    assert adapter.get_active_version("prompt", "mlops") == version


def test_set_active_version_raises_for_unregistered_name(adapter) -> None:
    with pytest.raises(ValueError, match="has no version"):
        adapter.set_active_version("prompt", "unknown", "1")


def test_set_active_version_raises_for_unregistered_version(adapter) -> None:
    adapter.register_version("prompt", "mlops", {"content": "v1"})
    with pytest.raises(ValueError, match="no version '2'"):
        adapter.set_active_version("prompt", "mlops", "2")


def test_get_version_raises_for_unknown_version(adapter) -> None:
    adapter.register_version("prompt", "mlops", {"content": "v1"})
    with pytest.raises(ValueError, match="no version '9'"):
        adapter.get_version("prompt", "mlops", "9")


def test_list_versions_returns_all_registered_versions(adapter) -> None:
    adapter.register_version("prompt", "mlops", {"content": "v1"})
    adapter.register_version("prompt", "mlops", {"content": "v2"})
    versions = adapter.list_versions("prompt", "mlops")
    assert set(versions) == {"1", "2"}


def test_list_names_returns_every_registered_name_for_a_kind(adapter) -> None:
    adapter.register_version("prompt", "mlops", {"content": "a"})
    adapter.register_version("prompt", "k8s", {"content": "b"})
    assert set(adapter.list_names("prompt")) == {"mlops", "k8s"}


def test_list_names_returns_empty_list_for_unknown_kind(adapter) -> None:
    assert adapter.list_names("rag-index") == []


def test_state_persists_across_separate_adapter_instances(adapter) -> None:
    adapter.register_version("prompt", "mlops", {"content": "v1"})
    reloaded = QdrantVersionRegistryAdapter(collection=adapter.collection)
    assert reloaded.list_versions("prompt", "mlops") == {"1": {"content": "v1"}}


def test_set_active_version_is_isolated_per_environment(adapter) -> None:
    adapter.register_version("prompt", "mlops", {"content": "v1"})
    adapter.register_version("prompt", "mlops", {"content": "v2"})
    adapter.set_active_version("prompt", "mlops", "1", environment="production")
    adapter.set_active_version("prompt", "mlops", "2", environment="development")

    assert adapter.get_active_version("prompt", "mlops", environment="production") == "1"
    assert adapter.get_active_version("prompt", "mlops", environment="development") == "2"
    # No environment argument means production, unchanged from before
    # environments existed — every pre-existing caller (chat.py,
    # ai-observability-server) keeps reading exactly what it read before.
    assert adapter.get_active_version("prompt", "mlops") == "1"


def test_get_active_version_returns_none_for_an_environment_never_activated(adapter) -> None:
    adapter.register_version("prompt", "mlops", {"content": "v1"})
    adapter.set_active_version("prompt", "mlops", "1", environment="production")

    assert adapter.get_active_version("prompt", "mlops", environment="staging") is None
