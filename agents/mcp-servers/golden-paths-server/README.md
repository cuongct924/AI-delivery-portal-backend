# golden-paths-server

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
  the caller's token to carry the `golden-paths:mutate` scope
  (`server.py::_require_confirmed_mutation`) — a caller hitting this
  server's endpoint directly, bypassing `chat.py` entirely, cannot execute
  either tool with just a plain "confirm=True" and no token to back it up.

## Auth

**Outbound** — calling `orchestration-api`'s `/prompts`/`/rag` endpoints:
this server authenticates as its own Thunder service-account
(`golden-paths-agent`, client_credentials, see `thunder_client.py`), not a
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
`services/orchestration-api/mcp_auth_client.py`) when it connects.

Known gap: `golden-paths-agent` and `orchestration-api-agent` both still
need registering in Thunder by hand, and `orchestration-api-agent`'s token
needs the `golden-paths:mutate` scope for `activate_prompt`/`rag_activate`
to work at all. Both are IdP-side provisioning steps outside this repo's
code.

Transport: `streamable-http`, `MCP_HOST`/`MCP_PORT` (default `0.0.0.0:9002`)
— discovered via the Backstage Catalog, not a hardcoded path.

## Run locally

```bash
bash scripts/run-mcp-local.sh golden-paths
```
