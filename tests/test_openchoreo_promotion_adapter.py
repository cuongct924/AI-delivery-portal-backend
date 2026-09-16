"""Tests adapters/openchoreo_promotion_adapter.py.

Same kubeconfig-mocking convention as tests/test_openchoreo_inference_adapter.py
— OpenChoreoPromotionAdapter.__init__ also calls
kubernetes.config.load_kube_config() eagerly.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.exceptions import ApiException

from adapters.openchoreo_promotion_adapter import (
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
        patch("adapters.openchoreo_promotion_adapter.config.load_kube_config"),
        patch(
            "adapters.openchoreo_promotion_adapter.client.CustomObjectsApi",
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


def test_promote_rejects_an_unknown_target_environment(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    with pytest.raises(ValueError, match="isn't a valid promotion target"):
        adapter.promote("development")
    mock_api.get_namespaced_custom_object.assert_not_called()


def test_promote_raises_when_source_environment_has_no_release(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object.side_effect = _not_found()

    with pytest.raises(ValueError, match="no release bound in 'development'"):
        adapter.promote("staging")


def test_promote_patches_an_existing_target_binding(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    # _upsert_binding now reads the target's current binding first (to know
    # what to record as "previous") before patching — one extra
    # get_namespaced_custom_object call compared to before that existed.
    mock_api.get_namespaced_custom_object.side_effect = [
        {"spec": {"projectRelease": "rel-1"}},  # source (development) lookup, in promote()
        {"spec": {"projectRelease": "rel-1"}},  # current staging, inside _upsert_binding
        {"spec": {"projectRelease": "rel-1"}},  # development re-read, in get_promotion_status
        {"spec": {"projectRelease": "rel-1"}},  # staging re-read, in get_promotion_status
        _not_found(),  # production re-read, in get_promotion_status
    ]

    adapter.promote("staging")

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


def test_promote_records_the_previous_release_as_an_annotation_when_it_changes(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object.side_effect = [
        {"spec": {"projectRelease": "rel-2"}},  # source (development) — new value
        {"spec": {"projectRelease": "rel-1"}},  # current staging — different!
        {"spec": {"projectRelease": "rel-2"}},  # development re-read
        {"spec": {"projectRelease": "rel-2"}},  # staging re-read
        _not_found(),  # production re-read
    ]

    adapter.promote("staging")

    body = mock_api.patch_namespaced_custom_object.call_args[0][-1]
    assert body["metadata"]["annotations"] == {_PREVIOUS_RELEASE_ANNOTATION: "rel-1"}


def test_promote_creates_the_target_binding_when_it_does_not_exist_yet(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object.side_effect = [
        {"spec": {"projectRelease": "rel-1"}},  # source (development) lookup
        _not_found(),  # current staging, inside _upsert_binding — doesn't exist yet
        {"spec": {"projectRelease": "rel-1"}},  # development re-read
        _not_found(),  # staging re-read (still not-found — this is a mock)
        _not_found(),  # production re-read
    ]
    mock_api.patch_namespaced_custom_object.side_effect = _not_found()

    status = adapter.promote("staging")

    mock_api.create_namespaced_custom_object.assert_called_once()
    assert status["project"] == "telco-fraud-detection"


def test_rollback_promotion_raises_when_nothing_bound_yet(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object.side_effect = _not_found()

    with pytest.raises(ValueError, match="no release bound in 'staging'"):
        adapter.rollback_promotion("staging")


def test_rollback_promotion_raises_when_no_previous_release_recorded(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object.return_value = {
        "spec": {"projectRelease": "rel-1"},
        "metadata": {},
    }

    with pytest.raises(ValueError, match="no prior release recorded"):
        adapter.rollback_promotion("staging")


def test_rollback_promotion_swaps_back_to_the_previous_release(
    adapter: OpenChoreoPromotionAdapter, mock_api: MagicMock
) -> None:
    bound_with_history = {
        "spec": {"projectRelease": "rel-2"},
        "metadata": {"annotations": {_PREVIOUS_RELEASE_ANNOTATION: "rel-1"}},
    }
    mock_api.get_namespaced_custom_object.side_effect = [
        bound_with_history,  # rollback_promotion's own read of "staging"
        bound_with_history,  # current staging, inside _upsert_binding
        {"spec": {"projectRelease": "rel-1"}},  # development re-read
        {"spec": {"projectRelease": "rel-1"}},  # staging re-read — now rolled back
        _not_found(),  # production re-read
    ]

    status = adapter.rollback_promotion("staging")

    assert status["environments"]["staging"] == "rel-1"
    mock_api.patch_namespaced_custom_object.assert_called_once()
