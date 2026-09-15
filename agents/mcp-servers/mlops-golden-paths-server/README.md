# mlops-golden-paths-server

MCP server for the MLOps Lifecycle Golden Path actions — Train->Track->
Register, Register->Deploy, and Serving LLM. Each tool is a thin HTTP
client into `orchestration-api`'s `/datasets`, `/trigger-training`,
`/models`, `/deploy-model`, and `/llm-deploy` endpoints — no business
logic lives here. Covers a different domain than
`agents/mcp-servers/llmops-golden-paths-server/` (LLMOps — prompt/RAG lifecycle);
kept as a separate server rather than folded into that one so each domain
has its own Thunder client identity, its own mutate scope, and its own
independently deployable/revocable blast radius — not because the code
itself couldn't share a process.

## Tools

- `validate_dataset` — auto-executable (read-only data-quality check, no
  mutation). Callable by any caller presenting a valid Thunder Bearer
  token (see Auth below) — no extra scope required.
- `trigger_training`, `register_model`, `prepare_deploy`,
  `prepare_llm_deploy` — tagged `destructive_hint=True`: real compute
  spend, a new registered model version, or production traffic changes.
  Never auto-called by `chat.py`'s tool loop, which must propose the call
  and wait for a human to approve it (`ChatResponse.pending_tool_call` /
  `ChatRequest.confirmed_tool_call`).

  The server itself also enforces this, not just `chat.py`: all 4 take a
  `confirm: bool = False` parameter (rejected unless `True`) and require
  the caller's token to carry the `mlops-golden-paths:mutate` scope
  (`server.py::_require_confirmed_mutation`) — a caller hitting this
  server's endpoint directly, bypassing `chat.py` entirely, cannot execute
  any of them with just a plain "confirm=True" and no token to back it up.

## Auth

**Outbound** — calling `orchestration-api`'s `/datasets`, `/trigger-training`,
`/models`, `/deploy-model`, `/llm-deploy` endpoints: this server
authenticates as its own Thunder service-account (`mlops-golden-paths-agent`,
client_credentials, see `thunder_client.py`), not a self-reported header.
`orchestration-api` verifies any presented token regardless of its own
`AUTH_ENABLED` — only a caller with no token falls back to the dev-bypass
identity. Note `POST /models/register` itself has no `Depends(get_current_user)`
(it's meant to be called from inside an Argo workflow pod) — sending it an
auth header anyway is harmless, just unused there.

**Inbound** — this server's own MCP endpoint (`/mcp`): every tool call
requires a valid Thunder Bearer token (`token_verifier.py`, wired via
`AuthSettings`/`token_verifier=` in `server.py`) — this is the server's
own gate, checked in-process, not something it trusts `orchestration-api`'s
`chat.py` to have already done. `orchestration-api` presents its own
Thunder identity (`orchestration-api-agent`, see
`services/orchestration-api/mcp_auth_client.py`, which requests both this
server's `mlops-golden-paths:mutate` scope and llmops-golden-paths-server's
`golden-paths:mutate` scope on every token it fetches — one shared
outbound identity, two independent servers to present it to).

`token_verifier.py` checks the token's `aud` against
`THUNDER_MCP_ALLOWED_CLIENTS` (default: `orchestration-api-agent`) as a
membership test, not equality against one fixed resource string —
Thunder's client_credentials tokens carry `aud == client_id`, there's no
separate "resource" audience to mint against (see llmops-golden-paths-server's
README for how this was verified empirically).

**Known gap — manual, IdP-side, still not done:** `mlops-golden-paths-agent`
and `orchestration-api-agent` both need registering as Thunder applications
(client_credentials grant) before any of this actually authenticates — see
llmops-golden-paths-server/README.md's own "Known gap" section; the same
registration blocker applies here (Thunder's admin API wasn't reachable
with any credential available at the time these servers were built).

Transport: `streamable-http`, `MCP_HOST`/`MCP_PORT` (default `0.0.0.0:9004`)
— discovered via the Backstage Catalog, not a hardcoded path.

## Run locally

```bash
pip install -q -r agents/mcp-servers/mlops-golden-paths-server/requirements.txt
PYTHONPATH=. python agents/mcp-servers/mlops-golden-paths-server/server.py
```
