"""Thunder client_credentials token fetch/cache — this service's own
identity when it connects out to an MCP server (mcp_client.py), mirroring
`agents/mcp-servers/golden-paths-server/thunder_client.py`'s pattern
one level up the call chain.

The `orchestration-api-agent` client still needs registering by hand in
Thunder (same prerequisite thunder_client.py notes for golden-paths-agent)
before an MCP server that enforces auth will accept this token.

Requests `MUTATE_SCOPE` on every token — verified empirically against a
live Thunder instance that it grants whatever scope a client_credentials
request asks for (no per-application scope allowlist enforced), so this is
what actually gets `golden-paths-server`'s `activate_prompt`/`rag_activate`
scope check to pass; omitting `scope=` from the request yields a token
with no `scope` claim at all.
"""

import os
import time
from typing import Final

import httpx
from core.config import settings

THUNDER_CLIENT_ID: Final[str] = os.getenv("MCP_CLIENT_ID", "orchestration-api-agent")
THUNDER_CLIENT_SECRET: Final[str] = os.getenv(
    "MCP_CLIENT_SECRET", "orchestration-api-agent-dev-secret"
)
# Must match golden-paths-server/server.py's MUTATE_SCOPE.
MUTATE_SCOPE: Final[str] = "golden-paths:mutate"

_cached_token: str | None = None
_cached_expiry: float = 0.0


def get_access_token() -> str:
    """Return a cached client_credentials access token, refreshing it once expired."""
    global _cached_token, _cached_expiry
    now = time.monotonic()
    if _cached_token is not None and now < _cached_expiry:
        return _cached_token

    response = httpx.post(
        f"{settings.thunder_url}/oauth2/token",
        data={
            "grant_type": "client_credentials",
            "client_id": THUNDER_CLIENT_ID,
            "client_secret": THUNDER_CLIENT_SECRET,
            "scope": MUTATE_SCOPE,
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
