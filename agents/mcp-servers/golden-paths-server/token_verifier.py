"""Verifies inbound Bearer tokens against Thunder's JWKS — this server's own
gate, not a rubber stamp for whatever `orchestration-api`/`chat.py` claims
was confirmed. Mirrors the verification shape in
`services/orchestration-api/auth/thunder.py::get_current_user` (JWKS fetch +
decode); duplicated rather than imported because this is a separate
deployable service with no dependency on orchestration-api's package.

Uses `pyjwt` + `cryptography` rather than `python-jose` (used by
orchestration-api) specifically because both are already transitive
dependencies of `mcp[cli]` here (it uses PyJWT internally for its own
bearer-auth support) — adding python-jose would mean a new pinned
dependency in this service's lock file for no benefit.
"""

import json
import logging
import os
from functools import lru_cache
from typing import Final, cast

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from jwt.algorithms import RSAAlgorithm
from mcp.server.auth.provider import AccessToken, TokenVerifier

THUNDER_URL: Final[str] = os.getenv("THUNDER_URL", "http://thunder.openchoreo.localhost:8080")
# Distinct from orchestration-api's own `THUNDER_AUDIENCE` (openchoreo-backstage-client) —
# tokens presented here must have been minted for *this* resource, not replayed
# from a different audience.
THUNDER_AUDIENCE: Final[str] = os.getenv("THUNDER_MCP_AUDIENCE", "golden-paths-server")

logger = logging.getLogger("golden_paths_server.token_verifier")


@lru_cache
def _jwks() -> dict:
    response = httpx.get(f"{THUNDER_URL}/oauth2/jwks", timeout=5)
    response.raise_for_status()
    return response.json()


def _signing_key_for(token: str) -> RSAPublicKey | None:
    """JWKS entries are always public keys (published for verification) —
    `from_jwk`'s return type is a broader union covering private keys too,
    since it's also used for signing; cast to the narrower public-key type
    `jwt.decode` actually accepts.
    """
    kid = jwt.get_unverified_header(token).get("kid")
    for key in _jwks().get("keys", []):
        if key.get("kid") == kid:
            return cast(RSAPublicKey, RSAAlgorithm.from_jwk(json.dumps(key)))
    return None


class ThunderTokenVerifier(TokenVerifier):
    """Verifies a Bearer token against Thunder's JWKS. Returns `None` on any
    failure — the SDK maps that to 401, never raises past this boundary."""

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            signing_key = _signing_key_for(token)
            if signing_key is None:
                return None
            claims = jwt.decode(token, signing_key, algorithms=["RS256"], audience=THUNDER_AUDIENCE)
        except Exception:
            logger.warning("rejected MCP bearer token", exc_info=True)
            return None

        client_id = claims.get("azp") or claims.get("sub")
        if not client_id:
            return None
        scope_claim = claims.get("scope") or ""
        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=scope_claim.split() if isinstance(scope_claim, str) else [],
            expires_at=claims.get("exp"),
            subject=claims.get("sub"),
        )
