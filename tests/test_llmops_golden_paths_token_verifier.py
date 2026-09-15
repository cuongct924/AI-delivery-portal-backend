"""agents/mcp-servers/llmops-golden-paths-server/token_verifier.py.

Loaded via conftest.load_module_from_path (not a bare
`from token_verifier import ...`) because mlops-golden-paths-server has
its own same-named token_verifier.py — see that helper's docstring.
mlops-golden-paths-server's copy is functionally identical (differs only
in its logger name), so this suite isn't duplicated for it.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import jwt
import pytest
from conftest import load_module_from_path
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

_SERVER_DIR = Path(__file__).parent.parent / "agents/mcp-servers/llmops-golden-paths-server"
token_verifier = load_module_from_path(
    "llmops_golden_paths_token_verifier", _SERVER_DIR / "token_verifier.py"
)

_PRIVATE_KEY: RSAPrivateKey = rsa.generate_private_key(public_exponent=65537, key_size=2048)
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
    token_verifier._jwks.cache_clear()
    yield
    token_verifier._jwks.cache_clear()


def _real_thunder_claims(client_id: str, scope: str | None = None) -> dict:
    """Shape empirically observed from a live Thunder instance's
    client_credentials grant: `aud` == `client_id` (no separate resource
    audience), no `azp` claim, `scope` present only if explicitly
    requested — not `azp`-driven like a typical OIDC IdP."""
    claims = {
        "aud": client_id,
        "client_id": client_id,
        "sub": client_id,
        "grant_type": "client_credentials",
        "exp": int(time.time()) + 300,
    }
    if scope is not None:
        claims["scope"] = scope
    return claims


@pytest.mark.asyncio
async def test_verify_token_accepts_allowed_client_in_allowlist() -> None:
    token = _sign(
        _real_thunder_claims("orchestration-api-agent", scope="llmops-golden-paths:mutate")
    )

    with patch("llmops_golden_paths_token_verifier.httpx.get", return_value=_mock_jwks_response()):
        access_token = await token_verifier.ThunderTokenVerifier().verify_token(token)

    assert access_token is not None
    assert access_token.client_id == "orchestration-api-agent"
    assert access_token.scopes == ["llmops-golden-paths:mutate"]


@pytest.mark.asyncio
async def test_verify_token_defaults_to_no_scopes_when_none_requested() -> None:
    # Real Thunder behavior: omitting scope= on the token request yields no
    # scope claim at all, not an empty string.
    token = _sign(_real_thunder_claims("orchestration-api-agent"))

    with patch("llmops_golden_paths_token_verifier.httpx.get", return_value=_mock_jwks_response()):
        access_token = await token_verifier.ThunderTokenVerifier().verify_token(token)

    assert access_token is not None
    assert access_token.scopes == []


@pytest.mark.asyncio
async def test_verify_token_rejects_client_not_in_allowlist() -> None:
    # A well-formed, correctly-signed token for some OTHER Thunder client
    # (e.g. llmops-golden-paths-agent's own outbound identity, or any
    # unrelated app) must not be accepted here — this server only expects
    # orchestration-api-agent to call it.
    token = _sign(_real_thunder_claims("some-other-client"))

    with patch("llmops_golden_paths_token_verifier.httpx.get", return_value=_mock_jwks_response()):
        access_token = await token_verifier.ThunderTokenVerifier().verify_token(token)

    assert access_token is None


@pytest.mark.asyncio
async def test_verify_token_respects_custom_allowlist() -> None:
    token = _sign(_real_thunder_claims("some-other-client"))

    with (
        patch("llmops_golden_paths_token_verifier.httpx.get", return_value=_mock_jwks_response()),
        patch(
            "llmops_golden_paths_token_verifier.ALLOWED_CLIENT_IDS",
            frozenset({"some-other-client"}),
        ),
    ):
        access_token = await token_verifier.ThunderTokenVerifier().verify_token(token)

    assert access_token is not None
    assert access_token.client_id == "some-other-client"


@pytest.mark.asyncio
async def test_verify_token_rejects_expired_token() -> None:
    claims = _real_thunder_claims("orchestration-api-agent")
    claims["exp"] = int(time.time()) - 10
    token = _sign(claims)

    with patch("llmops_golden_paths_token_verifier.httpx.get", return_value=_mock_jwks_response()):
        access_token = await token_verifier.ThunderTokenVerifier().verify_token(token)

    assert access_token is None


@pytest.mark.asyncio
async def test_verify_token_rejects_unknown_kid() -> None:
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = jwt.encode(
        _real_thunder_claims("orchestration-api-agent"),
        other_key,
        algorithm="RS256",
        headers={"kid": "not-in-jwks"},
    )
    with patch("llmops_golden_paths_token_verifier.httpx.get", return_value=_mock_jwks_response()):
        access_token = await token_verifier.ThunderTokenVerifier().verify_token(token)

    assert access_token is None
