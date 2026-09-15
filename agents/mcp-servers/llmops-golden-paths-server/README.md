# llmops-golden-paths-server

MCP server for the LLMOps Lifecycle Golden Path actions. Each tool is a
thin HTTP client into `orchestration-api`'s `/prompts` and `/rag` endpoints
— no business logic lives here.

## Tools

- `draft_prompt`, `evaluate_prompt`, `rag_ingest`, `rag_evaluate` — auto-executable.
  Callable by any caller presenting a valid Thunder Bearer token (see Auth
  below) — no extra scope required.
- `activate_prompt`, `rag_activate` — tagged `destructive_hint=True`. LLMOps
  activation has no PR-gate, so these are never auto-called by `chat.py`'s
  tool loop, which must propose the call and wait for a human to approve it
  (`ChatResponse.pending_tool_call` / `ChatRequest.confirmed_tool_call`).
  The server itself also enforces this, not just `chat.py`: both tools take
  a `confirm: bool = False` parameter (rejected unless `True`) and require
  the caller's token to carry the `llmops-golden-paths:mutate` scope
  (`server.py::_require_confirmed_mutation`) — a caller hitting this
  server's endpoint directly, bypassing `chat.py` entirely, cannot execute
  either tool with just a plain "confirm=True" and no token to back it up.

## Auth

**Outbound** — calling `orchestration-api`'s `/prompts`/`/rag` endpoints:
this server authenticates as its own Thunder service-account
(`llmops-golden-paths-agent`, client_credentials, see `thunder_client.py`), not a
self-reported header. `orchestration-api` verifies any presented token
regardless of its own `AUTH_ENABLED` — only a caller with no token falls
back to the dev-bypass identity, so Backstage Scaffolder (which sends no
token) is unaffected.

**Inbound** — this server's own MCP endpoint (`/mcp`): every tool call
requires a valid Thunder Bearer token (`token_verifier.py`, wired via
`AuthSettings`/`token_verifier=` in `server.py`) — this is the server's own
gate, checked in-process, not something it trusts `orchestration-api`'s
`chat.py` to have already done. `orchestration-api` presents its own
Thunder identity (`orchestration-api-agent`, see
`services/orchestration-api/mcp_auth_client.py`, which also requests the
`llmops-golden-paths:mutate` scope on every token it fetches).

`token_verifier.py` checks the token's `aud` against
`THUNDER_MCP_ALLOWED_CLIENTS` (default: `orchestration-api-agent`) as a
membership test, not equality against one fixed resource string —
verified empirically against a live Thunder instance that its
client_credentials tokens carry `aud == client_id`, there's no separate
"resource" audience to mint against.

**Known gap — manual, IdP-side, still not done:** `llmops-golden-paths-agent` and
`orchestration-api-agent` both need registering as Thunder applications
(client_credentials grant) before any of this actually authenticates.
Attempted this directly against the local k3d cluster
(`k3d-openchoreo-quick-start`, `thunder` namespace) and hit a hard wall:
Thunder's `/applications` management API returns 401/403 for every
approach tried — unauthenticated (works only from inside the one-time
Helm bootstrap hook, already run and gone), as an existing app's own
client_credentials token (`openchoreo-system-app` → 403 forbidden, no
management scope), and via `kubectl exec`+`kubectl port-forward` straight
to the pod (still 401). Registering these 2 clients needs either an admin
credential this session doesn't have, or another Helm-hook-style bootstrap
run — someone with real admin access to this Thunder instance needs to add
them, e.g. by extending the `thunder-bootstrap` Helm chart's ConfigMap
with 2 more scripts following the exact pattern already used by
`53-rca-agent-client.sh`/`57-service-mcp-app.sh` in that ConfigMap
(`client_credentials` grant, `token_endpoint_auth_method: client_secret_basic`,
`client_id`/`client_secret` matching the defaults in `thunder_client.py`/
`mcp_auth_client.py`) and re-running that hook.

Transport: `streamable-http`, `MCP_HOST`/`MCP_PORT` (default `0.0.0.0:9002`)
— discovered via the Backstage Catalog, not a hardcoded path.

## Run locally

```bash
bash scripts/run-mcp-local.sh llmops-golden-paths
```
