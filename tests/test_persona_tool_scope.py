"""persona_tool_scope.py — deny-by-default persona -> allowed MCP tool set."""

from persona_tool_scope import allowed_tools_for


def test_mlops_persona_gets_observability_and_golden_path_tools() -> None:
    allowed = allowed_tools_for("mlops")

    assert "get_model_metrics" in allowed
    assert "draft_prompt" in allowed
    assert "activate_prompt" in allowed
    assert "rag_activate" in allowed


def test_unregistered_persona_gets_no_tools() -> None:
    # "k8s" was removed (see persona_tool_scope.py's module docstring) —
    # kept in this list so a re-add doesn't slip back in unnoticed.
    for persona in (
        "k8s",
        "demo-persona",
        "agent-demo-persona",
        "registry-demo-persona",
        "unknown",
    ):
        assert allowed_tools_for(persona) == frozenset()
