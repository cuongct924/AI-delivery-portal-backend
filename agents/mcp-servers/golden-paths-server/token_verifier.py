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

Audience check is an allowlist of `client_id`s, not a fixed resource
string: verified empirically against a live Thunder instance (k3d
`openchoreo-quick-start`, `oauth2/token` for an existing app) that its
client_credentials tokens always carry `aud == client_id` — there is no
separate "resource" audience to mint against. `services/orchestration-api
/auth/thunder.py`'s `thunder_audience = "openchoreo-backstage-client"`
setting works the same way (that value equals Backstage's own client_id,
its only allowed caller); this generalizes that pattern to a set, since
this server expects one specific caller (`orchestration-api-agent`), not
Backstage.
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
# client_ids allowed to call this server's MCP endpoint at all (comma-separated).
# Thunder's client_credentials tokens carry aud == the caller's own client_id
# (empirically confirmed, see module docstring) — so this is checked as
# membership, not equality against one fixed string.
ALLOWED_CLIENT_IDS: Final[frozenset[str]] = frozenset(
    c.strip()
    for c in os.getenv("THUNDER_MCP_ALLOWED_CLIENTS", "orchestration-api-agent").split(",")
    if c.strip()
)

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
            # verify_aud=False: we check `aud` against an allowlist of
            # client_ids ourselves below, not equality against one fixed
            # string — PyJWT's built-in `audience=` only supports the latter.
            claims = jwt.decode(
                token, signing_key, algorithms=["RS256"], options={"verify_aud": False}
            )
        except Exception:
            logger.warning("rejected MCP bearer token", exc_info=True)
            return None

        client_id = claims.get("azp") or claims.get("client_id") or claims.get("sub")
        if not client_id:
            return None
        aud = claims.get("aud")
        aud_values = {aud} if isinstance(aud, str) else set(aud) if aud else set()
        if not aud_values & ALLOWED_CLIENT_IDS:
            logger.warning("rejected MCP bearer token: aud %r not in allowlist", aud)
            return None
        scope_claim = claims.get("scope") or ""
        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=scope_claim.split() if isinstance(scope_claim, str) else [],
            expires_at=claims.get("exp"),
            subject=claims.get("sub"),
        )
