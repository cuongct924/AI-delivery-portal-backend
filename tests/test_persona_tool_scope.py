"""persona_tool_scope.py — deny-by-default persona -> allowed MCP tool set."""

from persona_tool_scope import PERSONA_ALLOWED_TOOLS, allowed_tools_for


def test_k8s_persona_is_read_only() -> None:
    allowed = allowed_tools_for("k8s")

    assert "get_model_metrics" in allowed
    assert "activate_prompt" not in allowed
    assert "rag_activate" not in allowed


def test_mlops_persona_gets_observability_and_golden_path_tools() -> None:
    allowed = allowed_tools_for("mlops")

    assert "get_model_metrics" in allowed
    assert "draft_prompt" in allowed
    assert "activate_prompt" in allowed
    assert "rag_activate" in allowed


def test_unregistered_persona_gets_no_tools() -> None:
    for persona in ("demo-persona", "agent-demo-persona", "registry-demo-persona", "unknown"):
        assert allowed_tools_for(persona) == frozenset()


def test_persona_map_has_no_overlap_bugs_between_registered_personas() -> None:
    # Every tool granted to k8s must also be granted to mlops (mlops is a
    # superset) — catches an accidental typo splitting one persona's set
    # from the other's.
    assert PERSONA_ALLOWED_TOOLS["k8s"] <= PERSONA_ALLOWED_TOOLS["mlops"]
