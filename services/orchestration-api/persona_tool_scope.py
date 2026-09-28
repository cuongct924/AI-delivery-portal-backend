"""Which MCP tools a chat persona may see/call — chat.py filters
`registry.list_tools()` through this before handing the tool list to the
LLM, so a persona can't be steered into calling a destructive tool the
model was never even offered.

A persona absent from `PERSONA_ALLOWED_TOOLS` gets no tools at all — this
is a deny-by-default allowlist, not a denylist. Tool names must match the
`@mcp.tool` function names in the corresponding server exactly.

The "k8s" persona (Kubernetes read-only ops) was removed — OpenChoreo's own
built-in MCP server now covers pod/log/event lookups, so a separate
Portal-native persona for it duplicated coverage the platform already
provides.
"""

type ToolScope = frozenset[str]

# agents/mcp-servers/ai-observability-server/server.py — read-only.
_AI_OBSERVABILITY_TOOLS: ToolScope = frozenset(
    {
        "list_experiments",
        "get_model_metrics",
        "check_model_latency",
        "get_llm_spend",
        "get_active_prompt_version",
        "get_active_rag_version",
        "get_eval_score_trend",
    }
)

# agents/mcp-servers/llmops-golden-paths-server/server.py — LLMOps prompt/
# RAG lifecycle, includes the two NEEDS_CONFIRMATION mutating tools.
_GOLDEN_PATH_TOOLS: ToolScope = frozenset(
    {
        "draft_prompt",
        "evaluate_prompt",
        "activate_prompt",
        "rag_ingest",
        "rag_evaluate",
        "rag_activate",
    }
)

# agents/mcp-servers/mlops-golden-paths-server/server.py — Train->Register->
# Deploy/Serving LLM lifecycle, includes the 4 NEEDS_CONFIRMATION tools.
_MLOPS_GOLDEN_PATH_TOOLS: ToolScope = frozenset(
    {
        "validate_dataset",
        "trigger_training",
        "register_model",
        "prepare_deploy",
        "prepare_llm_deploy",
    }
)

PERSONA_ALLOWED_TOOLS: dict[str, ToolScope] = {
    "mlops": _AI_OBSERVABILITY_TOOLS | _GOLDEN_PATH_TOOLS | _MLOPS_GOLDEN_PATH_TOOLS,
}


def allowed_tools_for(persona: str) -> ToolScope:
    """Deny-by-default: a persona not registered above gets no tools."""
    return PERSONA_ALLOWED_TOOLS.get(persona, frozenset())


# agents/mcp-servers/golden-path-guide-server/server.py — the only server
# routers/portal_assistant.py's system prompt ever tells the model to call
# (see routers/prompts.py's "mlops" persona content: every tool it names —
# list_golden_paths, get_golden_path_schema, propose_golden_path_draft,
# list_registered_models, list_rag_collections, list_prompts, list_secrets,
# search_huggingface_models — lives here). All read-only.
#
# Deliberately excludes ai-observability/llmops/mlops-golden-paths tools
# (chat.py's "mlops" persona scope above, which DOES execute the
# NEEDS_CONFIRMATION tools) — portal_assistant.py never calls those; its own
# docstring says a destructive tool is only ever described, never called,
# and the system prompt already tells the model their names in plain text
# ("Do NOT call an execution tool (trigger_training, register_model, ...)"),
# so omitting their JSON schemas here costs nothing: the model was never
# going to call them, and `result.pending`'s echo-back is a backstop for the
# model ignoring that instruction, not the primary UX path. Measured
# saving: 33 tools/~6.1k tok -> these 10/~1.5k tok, 75% off every round's
# tool-definition cost.
PORTAL_ASSISTANT_TOOLS: ToolScope = frozenset(
    {
        "list_golden_paths",
        "get_golden_path_guide",
        "get_golden_path_schema",
        "propose_golden_path_draft",
        "estimate_golden_path_cost",
        "list_registered_models",
        "list_rag_collections",
        "list_prompts",
        "list_secrets",
        "search_huggingface_models",
    }
)
