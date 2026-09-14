# golden-path-guide-server

MCP server for the "how do I run Golden Path X" domain. Both tools are
thin HTTP clients into `orchestration-api`'s `/golden-paths` endpoints —
no business logic lives here.

## Tools

- `list_golden_paths` — every mlops/llmops Golden Path template's name,
  title, description, tags.
- `get_golden_path_guide` — one template's full parameter list and step
  order, by `name`.

Both `read_only_hint=True` — never mutates state, no confirmation needed.

## Why not RAG

The mlops/llmops Golden Path templates are a small, fixed set (~8), each
with a short structured spec. `orchestration-api`'s `/golden-paths`
endpoints (`catalog_client.py`) read the live Backstage Catalog entity on
every call — always in sync with the real `template.yaml`, with no
ingest/re-ingest step to forget. A Qdrant-backed RAG index would add
staleness risk and infra for a corpus this size without a real benefit.

## Auth

No auth header sent — same as `observability-server`, since both are
read-only against routes that fall back to the dev-bypass identity when
`AUTH_ENABLED=false` (see `.env`). Give this server a Thunder
client_credentials identity (`thunder_client.py`, `golden-paths-server`'s
pattern) if that changes.

Transport: `streamable-http`, `MCP_HOST`/`MCP_PORT` (default `0.0.0.0:9003`)
— discovered via the Backstage Catalog, not a hardcoded path.

## Run locally

```bash
bash scripts/run-mcp-local.sh golden-path-guide
```
