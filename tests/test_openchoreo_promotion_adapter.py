"""Tests adapters/delivery/openchoreo_promotion_adapter.py.

Same kubeconfig-mocking convention as tests/test_openchoreo_inference_adapter.py
— OpenChoreoPromotionAdapter.__init__ also calls load_kube_config_once()
eagerly.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.exceptions import ApiException

from adapters.delivery.openchoreo_promotion_adapter import (
    _PREVIOUS_RELEASE_ANNOTATION,
    GROUP,
    PLURAL,
    VERSION,
    OpenChoreoPromotionAdapter,
)


@pytest.fixture
def mock_api() -> MagicMock:
    return MagicMock()


@pytest.fixture
def adapter(mock_api: MagicMock) -> Iterator[OpenChoreoPromotionAdapter]:
    with (
        patch("adapters.delivery.openchoreo_promotion_adapter.load_kube_config_once"),
        patch(
            "adapters.delivery.openchoreo_promotion_adapter.client.CustomObjectsApi",
            return_value=mock_api,
        ),
    ):
        yield OpenChoreoPromotionAdapter()


def _not_found() -> ApiException:
    return ApiException(status=404)


def test_get_promotion_status_reports_none_for_unbound_environments(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object.side_effect = [
        {"spec": {"projectRelease": "telco-fraud-detection-abc123"}},
        _not_found(),
        _not_found(),
    ]

    status = adapter.get_promotion_status()

    assert status["environments"] == {
        "development": "telco-fraud-detection-abc123",
        "staging": None,
        "production": None,
    }
    assert status["prod_pending_approval"] is False


def test_get_promotion_status_flags_pending_approval_when_staging_is_ahead(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object.side_effect = [
        {"spec": {"projectRelease": "rel-2"}},
        {"spec": {"projectRelease": "rel-2"}},
        {"spec": {"projectRelease": "rel-1"}},
    ]

    status = adapter.get_promotion_status()

    assert status["prod_pending_approval"] is True


def test_get_promotion_status_reraises_non_404_errors(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object.side_effect = ApiException(status=500)

    with pytest.raises(ApiException) as exc_info:
        adapter.get_promotion_status()
    assert exc_info.value.status == 500


def test_resolve_promotion_release_rejects_an_unknown_target_environment(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    with pytest.raises(ValueError, match="isn't a valid promotion target"):
        adapter.resolve_promotion_release("development")
    mock_api.get_namespaced_custom_object.assert_not_called()


def test_resolve_promotion_release_raises_when_source_environment_has_no_release(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object.side_effect = _not_found()

    with pytest.raises(ValueError, match="no release bound in 'development'"):
        adapter.resolve_promotion_release("staging")


def test_resolve_promotion_release_does_not_write_anything(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object.return_value = {"spec": {"projectRelease": "rel-1"}}

    release = adapter.resolve_promotion_release("staging")

    assert release == "rel-1"
    mock_api.patch_namespaced_custom_object.assert_not_called()
    mock_api.create_namespaced_custom_object.assert_not_called()


def test_confirm_promotion_patches_an_existing_target_binding(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    # _upsert_binding reads the target's current binding first (to know
    # what to record as "previous") before patching.
    mock_api.get_namespaced_custom_object.side_effect = [
        {"spec": {"projectRelease": "rel-1"}},  # current staging, inside _upsert_binding
        {"spec": {"projectRelease": "rel-1"}},  # development re-read, in get_promotion_status
        {"spec": {"projectRelease": "rel-1"}},  # staging re-read, in get_promotion_status
        _not_found(),  # production re-read, in get_promotion_status
    ]

    adapter.confirm_promotion("staging", "rel-1")

    # Same release as what's already bound — no annotation to record.
    mock_api.patch_namespaced_custom_object.assert_called_once_with(
        GROUP,
        VERSION,
        "default",
        PLURAL,
        "telco-fraud-detection-staging",
        {
            "apiVersion": f"{GROUP}/{VERSION}",
            "kind": "ProjectReleaseBinding",
            "metadata": {
                "name": "telco-fraud-detection-staging",
                "labels": {
                    "openchoreo.dev/project": "telco-fraud-detection",
                    "openchoreo.dev/environment": "staging",
                },
            },
            "spec": {
                "environment": "staging",
                "owner": {"projectName": "telco-fraud-detection"},
                "projectRelease": "rel-1",
            },
        },
    )
    mock_api.create_namespaced_custom_object.assert_not_called()


def test_confirm_promotion_records_the_previous_release_as_an_annotation_when_it_changes(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object.side_effect = [
        {"spec": {"projectRelease": "rel-1"}},  # current staging — different!
        {"spec": {"projectRelease": "rel-2"}},  # development re-read
        {"spec": {"projectRelease": "rel-2"}},  # staging re-read
        _not_found(),  # production re-read
    ]

    adapter.confirm_promotion("staging", "rel-2")

    body = mock_api.patch_namespaced_custom_object.call_args[0][-1]
    assert body["metadata"]["annotations"] == {_PREVIOUS_RELEASE_ANNOTATION: "rel-1"}


def test_confirm_promotion_creates_the_target_binding_when_it_does_not_exist_yet(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object.side_effect = [
        _not_found(),  # current staging, inside _upsert_binding — doesn't exist yet
        {"spec": {"projectRelease": "rel-1"}},  # development re-read
        _not_found(),  # staging re-read (still not-found — this is a mock)
        _not_found(),  # production re-read
    ]
    mock_api.patch_namespaced_custom_object.side_effect = _not_found()

    status = adapter.confirm_promotion("staging", "rel-1")

    mock_api.create_namespaced_custom_object.assert_called_once()
    assert status["project"] == "telco-fraud-detection"


def test_resolve_rollback_release_raises_when_nothing_bound_yet(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object.side_effect = _not_found()

    with pytest.raises(ValueError, match="no release bound in 'staging'"):
        adapter.resolve_rollback_release("staging")


def test_resolve_rollback_release_raises_when_no_previous_release_recorded(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object.return_value = {
        "spec": {"projectRelease": "rel-1"},
        "metadata": {},
    }

    with pytest.raises(ValueError, match="no prior release recorded"):
        adapter.resolve_rollback_release("staging")


def test_resolve_rollback_release_returns_the_previous_release(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object.return_value = {
        "spec": {"projectRelease": "rel-2"},
        "metadata": {"annotations": {_PREVIOUS_RELEASE_ANNOTATION: "rel-1"}},
    }

    release = adapter.resolve_rollback_release("staging")

    assert release == "rel-1"
    mock_api.patch_namespaced_custom_object.assert_not_called()


def test_confirm_promotion_applies_a_resolved_rollback(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    bound_with_history = {
        "spec": {"projectRelease": "rel-2"},
        "metadata": {"annotations": {_PREVIOUS_RELEASE_ANNOTATION: "rel-1"}},
    }
    mock_api.get_namespaced_custom_object.side_effect = [
        bound_with_history,  # current staging, inside _upsert_binding
        {"spec": {"projectRelease": "rel-1"}},  # development re-read
        {"spec": {"projectRelease": "rel-1"}},  # staging re-read — now rolled back
        _not_found(),  # production re-read
    ]

    status = adapter.confirm_promotion("staging", "rel-1")

    assert status["environments"]["staging"] == "rel-1"
    mock_api.patch_namespaced_custom_object.assert_called_once()
