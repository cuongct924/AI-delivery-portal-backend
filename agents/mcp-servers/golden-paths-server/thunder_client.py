"""Thunder client_credentials token fetch/cache — gives this server its own
verifiable service-account identity instead of a self-reported header.

The `golden-paths-agent` client still needs registering by hand in Thunder
— see infra/openchoreo/2.3-notes.md.
"""

import os
import time
from typing import Final

import httpx

THUNDER_URL: Final[str] = os.getenv("THUNDER_URL", "http://thunder.openchoreo.localhost:8080")
THUNDER_CLIENT_ID: Final[str] = os.getenv("THUNDER_CLIENT_ID", "golden-paths-agent")
THUNDER_CLIENT_SECRET: Final[str] = os.getenv(
    "THUNDER_CLIENT_SECRET", "golden-paths-agent-dev-secret"
)

_cached_token: str | None = None
_cached_expiry: float = 0.0


def get_access_token() -> str:
    """Return a cached client_credentials access token, refreshing it once expired."""
    global _cached_token, _cached_expiry
    now = time.monotonic()
    if _cached_token is not None and now < _cached_expiry:
        return _cached_token

    response = httpx.post(
        f"{THUNDER_URL}/oauth2/token",
        data={
            "grant_type": "client_credentials",
            "client_id": THUNDER_CLIENT_ID,
            "client_secret": THUNDER_CLIENT_SECRET,
        },
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    # Refresh 10s early so a token doesn't expire mid-request.
    _cached_expiry = now + max(payload["expires_in"] - 10, 0)
    token: str = payload["access_token"]
    _cached_token = token
    return token


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {get_access_token()}"}
