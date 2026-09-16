"""OIDC authentication via Thunder (OpenChoreo's bundled IdP) — every
request goes through here. `auth_enabled=False` makes auth optional, not
off: a token is verified for real if present, and only a request with no
token at all falls back to a fake dev user.
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


def user_has_role(user: dict, role: str) -> bool:
    """Checks a role/group claim on the dict get_current_user() returned.

    The local-dev bypass user (`sub == "local-dev"`, only reachable with
    `AUTH_ENABLED=false`) always has every role — same trust boundary as
    that bypass already crossing full authentication, not a separate
    privilege escalation. A real token's roles live under whichever of
    "roles"/"groups" Thunder's client actually populates (mirrors
    persona_tool_scope.py's allowlist-not-denylist stance: a claim shaped
    unexpectedly denies the role rather than silently granting it).
    """
    if user.get("sub") == "local-dev":
        return True
    roles = user.get("roles") or user.get("groups") or []
    return role in roles
