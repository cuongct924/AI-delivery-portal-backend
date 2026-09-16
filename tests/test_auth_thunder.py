"""services/orchestration-api/auth/thunder.py — patches `settings`/`jwt.decode`
directly, same pattern as tests/test_chat_router.py."""

from unittest.mock import patch

import pytest
from auth.thunder import get_current_user, user_has_role
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials


def _bearer(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_no_token_and_auth_disabled_returns_fake_dev_user() -> None:
    with patch("auth.thunder.settings") as mock_settings:
        mock_settings.auth_enabled = False
        user = get_current_user(credentials=None)
    assert user == {"sub": "local-dev", "preferred_username": "dev"}


def test_no_token_and_auth_enabled_raises_401() -> None:
    with patch("auth.thunder.settings") as mock_settings:
        mock_settings.auth_enabled = True
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(credentials=None)
    assert exc_info.value.status_code == 401


def test_valid_token_is_verified_even_when_auth_disabled() -> None:
    """A presented token is always verified, regardless of AUTH_ENABLED."""
    with (
        patch("auth.thunder.settings") as mock_settings,
        patch("auth.thunder._jwks", return_value={"keys": []}),
        patch("auth.thunder.jwt.decode", return_value={"azp": "golden-paths-agent"}),
    ):
        mock_settings.auth_enabled = False
        user = get_current_user(credentials=_bearer("a-real-token"))
    assert user == {"azp": "golden-paths-agent"}


def test_invalid_token_raises_401_even_when_auth_disabled() -> None:
    """A bad token must not silently fall back to the fake dev user."""
    with (
        patch("auth.thunder.settings") as mock_settings,
        patch("auth.thunder._jwks", return_value={"keys": []}),
        patch("auth.thunder.jwt.decode", side_effect=Exception("bad signature")),
    ):
        mock_settings.auth_enabled = False
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(credentials=_bearer("a-bad-token"))
    assert exc_info.value.status_code == 401


def test_user_has_role_local_dev_bypass_has_every_role() -> None:
    assert user_has_role({"sub": "local-dev", "preferred_username": "dev"}, "llm-ops-admin")


def test_user_has_role_checks_roles_claim() -> None:
    assert user_has_role({"sub": "real-user", "roles": ["llm-ops-admin"]}, "llm-ops-admin")
    assert not user_has_role({"sub": "real-user", "roles": ["viewer"]}, "llm-ops-admin")


def test_user_has_role_checks_groups_claim_when_roles_absent() -> None:
    assert user_has_role({"sub": "real-user", "groups": ["llm-ops-admin"]}, "llm-ops-admin")


def test_user_has_role_false_when_no_role_claim_present() -> None:
    assert not user_has_role({"sub": "real-user"}, "llm-ops-admin")
