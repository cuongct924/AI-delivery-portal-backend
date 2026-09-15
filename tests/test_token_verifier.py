"""agents/mcp-servers/golden-paths-server/token_verifier.py — importable
unqualified via pyproject.toml's `pythonpath`/`extraPaths`, same as this
server's own runtime (a standalone script, its directory implicitly on
sys.path)."""

import time
from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from token_verifier import THUNDER_AUDIENCE, ThunderTokenVerifier

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_KID = "test-key"


def _jwk() -> dict:
    from jwt.algorithms import RSAAlgorithm

    jwk = RSAAlgorithm.to_jwk(_PRIVATE_KEY.public_key(), as_dict=True)
    jwk["kid"] = _KID
    return jwk


def _sign(claims: dict) -> str:
    return jwt.encode(claims, _PRIVATE_KEY, algorithm="RS256", headers={"kid": _KID})


def _mock_jwks_response() -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"keys": [_jwk()]}
    response.raise_for_status.return_value = None
    return response


@pytest.fixture(autouse=True)
def _clear_jwks_cache():
    from token_verifier import _jwks

    _jwks.cache_clear()
    yield
    _jwks.cache_clear()


@pytest.mark.asyncio
async def test_verify_token_accepts_valid_signature_and_audience() -> None:
    token = _sign(
        {
            "sub": "user-1",
            "azp": "golden-paths-agent",
            "aud": THUNDER_AUDIENCE,
            "scope": "golden-paths:mutate",
            "exp": int(time.time()) + 300,
        }
    )
    with patch("token_verifier.httpx.get", return_value=_mock_jwks_response()):
        access_token = await ThunderTokenVerifier().verify_token(token)

    assert access_token is not None
    assert access_token.client_id == "golden-paths-agent"
    assert access_token.scopes == ["golden-paths:mutate"]


@pytest.mark.asyncio
async def test_verify_token_rejects_wrong_audience() -> None:
    token = _sign(
        {
            "sub": "user-1",
            "azp": "golden-paths-agent",
            "aud": "some-other-service",
            "exp": int(time.time()) + 300,
        }
    )
    with patch("token_verifier.httpx.get", return_value=_mock_jwks_response()):
        access_token = await ThunderTokenVerifier().verify_token(token)

    assert access_token is None


@pytest.mark.asyncio
async def test_verify_token_rejects_expired_token() -> None:
    token = _sign(
        {
            "sub": "user-1",
            "azp": "golden-paths-agent",
            "aud": THUNDER_AUDIENCE,
            "exp": int(time.time()) - 10,
        }
    )
    with patch("token_verifier.httpx.get", return_value=_mock_jwks_response()):
        access_token = await ThunderTokenVerifier().verify_token(token)

    assert access_token is None


@pytest.mark.asyncio
async def test_verify_token_rejects_unknown_kid() -> None:
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = jwt.encode(
        {"sub": "user-1", "aud": THUNDER_AUDIENCE, "exp": int(time.time()) + 300},
        other_key,
        algorithm="RS256",
        headers={"kid": "not-in-jwks"},
    )
    with patch("token_verifier.httpx.get", return_value=_mock_jwks_response()):
        access_token = await ThunderTokenVerifier().verify_token(token)

    assert access_token is None
