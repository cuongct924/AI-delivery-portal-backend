# golden-paths-server

MCP server for the LLMOps Lifecycle Golden Path actions. Each tool is a
thin HTTP client into `orchestration-api`'s `/prompts` and `/rag` endpoints
— no business logic lives here.

## Tools

- `draft_prompt`, `evaluate_prompt`, `rag_ingest`, `rag_evaluate` — auto-executable.
- `activate_prompt`, `rag_activate` — tagged `destructive_hint=True`. LLMOps
  activation has no PR-gate, so these are never auto-called — `chat.py`'s
  tool loop must propose and wait for confirmation.

## Auth

- This server authenticates as its own Thunder service-account
  (`golden-paths-agent`, client_credentials, see `thunder_client.py`),
  not a self-reported header. Thunder is OpenChoreo's bundled IdP —
  replaced Keycloak in Phase 2, sub-step 2.3.
- `orchestration-api` verifies any presented token regardless of its own
  `AUTH_ENABLED` — only a caller with no token falls back to the dev-bypass
  identity, so Backstage Scaffolder (which sends no token) is unaffected.
- Known gap: the `golden-paths-agent` client itself still needs
  registering in Thunder by hand (its admin console, or
  `/oauth2/dcr/register` with an authenticated admin caller — not yet
  automated) with an `orchestration-api` audience claim, same shape as the
  Keycloak client it replaces. See infra/openchoreo/2.3-notes.md.

Transport: `streamable-http`, `MCP_HOST`/`MCP_PORT` (default `0.0.0.0:9002`)
— discovered via the Backstage Catalog, not a hardcoded path.

## Run locally

```bash
bash scripts/run-mcp-local.sh golden-paths
```
