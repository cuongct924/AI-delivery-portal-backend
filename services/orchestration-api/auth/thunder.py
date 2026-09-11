"""OIDC authentication via Thunder (OpenChoreo's bundled IdP) — every
request goes through here. `auth_enabled=False` makes auth optional, not
off: a token is verified for real if present, and only a request with no
token at all falls back to a fake dev user. Set `AUTH_ENABLED=true` to
require a valid token always.

Replaces auth/keycloak.py (Phase 2, sub-step 2.3) — Thunder has no realm
concept, so its JWKS/token endpoints are fixed paths under `thunder_url`
instead of Keycloak's `/realms/<realm>/...` pattern (confirmed against
Thunder's own /.well-known/openid-configuration on the local
openchoreo-quick-start cluster: jwks_uri is `{base}/oauth2/jwks`, and
`client_credentials` is a supported grant_type).
"""

import logging
from functools import lru_cache

import httpx
from core.config import settings
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt

bearer_scheme = HTTPBearer(auto_error=False)
logger = logging.getLogger("orchestration_api.auth")


@lru_cache
def _jwks() -> dict:
    url = f"{settings.thunder_url}/oauth2/jwks"
    response = httpx.get(url, timeout=5)
    response.raise_for_status()
    return response.json()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    if credentials is None:
        if not settings.auth_enabled:
            logger.info("authenticated as local-dev (no token, AUTH_ENABLED=false)")
            return {"sub": "local-dev", "preferred_username": "dev"}
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing Bearer token")

    # A token was presented — always verify it, even under dev-bypass.
    try:
        claims = jwt.decode(
            credentials.credentials,
            _jwks(),
            algorithms=["RS256"],
            audience=settings.thunder_audience,
        )
    except Exception as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {exc}") from exc

    identity = claims.get("azp") or claims.get("preferred_username") or claims.get("sub")
    logger.info("authenticated as %s", identity)
    return claims
