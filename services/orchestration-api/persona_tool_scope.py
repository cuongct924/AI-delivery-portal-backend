"""Which MCP tools a chat persona may see/call — chat.py filters
`registry.list_tools()` through this before handing the tool list to the
LLM, so e.g. a "k8s" persona (self-described read-only in its prompt text)
can't be steered into calling a destructive tool the model was never even
offered.

A persona absent from `PERSONA_ALLOWED_TOOLS` gets no tools at all — this
is a deny-by-default allowlist, not a denylist. Tool names must match the
`@mcp.tool` function names in the corresponding server exactly.
"""

type ToolScope = frozenset[str]

# agents/mcp-servers/observability-server/server.py — read-only.
_OBSERVABILITY_TOOLS: ToolScope = frozenset(
    {
        "list_experiments",
        "get_model_metrics",
        "check_pod_status",
        "get_logs",
        "query_metric",
        "check_model_latency",
        "get_promotion_status",
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

# agents/mcp-servers/mlops-golden-paths-server/server.py — Train->Track->
# Register / Register->Deploy / Serving LLM lifecycle, includes the 4
# NEEDS_CONFIRMATION mutating tools.
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
    "k8s": _OBSERVABILITY_TOOLS,
    "mlops": _OBSERVABILITY_TOOLS | _GOLDEN_PATH_TOOLS | _MLOPS_GOLDEN_PATH_TOOLS,
}


def allowed_tools_for(persona: str) -> ToolScope:
    """Deny-by-default: a persona not registered above gets no tools."""
    return PERSONA_ALLOWED_TOOLS.get(persona, frozenset())
