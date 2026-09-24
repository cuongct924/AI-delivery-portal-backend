"""services/orchestration-api/routers/rag.py — patches the module-level
`llm_gateway_adapter`/`vector_store_adapter`/`registry_adapter` singletons
and `judge_response`/`evaluate_gate`, same pattern as
tests/test_models_router.py — calls route functions directly.
"""

from unittest.mock import patch

import pytest
from routers.rag import (
    RagActivateRequest,
    RagEvalCase,
    RagEvaluateRequest,
    RagIngestRequest,
    _chunk_text,
    _resolve_source_path,
    list_rag_collection_versions,
    list_rag_collections,
    list_rag_sources,
    rag_activate,
    rag_evaluate,
    rag_ingest,
)


def test_chunk_text_slides_with_overlap() -> None:
    assert _chunk_text("0123456789", chunk_size=4, chunk_overlap=1) == ["0123", "3456", "6789", "9"]


def test_resolve_source_path_returns_absolute_unchanged(tmp_path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("x")
    assert _resolve_source_path(str(doc)) == doc


def test_resolve_source_path_falls_back_to_repo_root(tmp_path, monkeypatch) -> None:
    # `make run-orchestration-api` runs with CWD=services/orchestration-api, so
    # a template's "docs/architecture-overview.md" must still resolve via the
    # __file__-derived repo root rather than only the process CWD.
    monkeypatch.chdir(tmp_path)
    resolved = _resolve_source_path("docs/architecture-overview.md")
    assert resolved.is_file()
    assert resolved.name == "architecture-overview.md"


def test_chunk_text_guards_against_zero_step() -> None:
    # chunk_overlap >= chunk_size would infinite-loop without the max(1, ...) guard.
    chunks = _chunk_text("0123456789", chunk_size=4, chunk_overlap=4)
    assert len(chunks) > 0


def test_rag_ingest_embeds_chunks_and_registers_version(tmp_path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("0123456789")
    request = RagIngestRequest(
        collection="smoke-test", source_paths=[str(doc)], chunk_size=4, chunk_overlap=1
    )
    with (
        patch("routers.rag.llm_gateway_adapter") as mock_gateway,
        patch("routers.rag.vector_store_adapter") as mock_vector_store,
        patch("routers.rag.registry_adapter") as mock_registry,
    ):
        mock_gateway.embed.return_value = [[0.1, 0.2]] * 4
        mock_registry.register_version.return_value = "1"
        response = rag_ingest(request)

    assert response.collection == "smoke-test"
    assert response.index_version == "1"
    assert response.chunks_ingested == 4
    mock_vector_store.ensure_collection.assert_called_once_with(
        vector_size=2, collection="smoke-test"
    )
    upsert_args = mock_vector_store.upsert.call_args
    assert len(upsert_args.args[0]) == 4  # ids
    assert upsert_args.args[1] == [[0.1, 0.2]] * 4  # vectors
    assert all(p["source"] == str(doc) for p in upsert_args.args[2])  # payloads
    # The version is registered first, then stamped onto every point so
    # search can filter to it.
    assert upsert_args.kwargs == {"collection": "smoke-test", "index_version": "1"}
    mock_registry.register_version.assert_called_once_with(
        "rag-index", "smoke-test", {"chunks_ingested": 4, "source_paths": [str(doc)]}
    )


def test_rag_evaluate_computes_pass_rate_and_forwards_model() -> None:
    request = RagEvaluateRequest(
        collection="smoke-test",
        index_version="1",
        eval_cases=[RagEvalCase(question="q1"), RagEvalCase(question="q2")],
        model="llama-3-8b-self-hosted",
    )
    with (
        patch("routers.rag.llm_gateway_adapter") as mock_gateway,
        patch("routers.rag.vector_store_adapter") as mock_vector_store,
        patch("routers.rag.judge_response") as mock_judge,
        patch("routers.rag.evaluate_gate") as mock_gate,
    ):
        mock_gateway.embed.return_value = [[0.1]]
        mock_vector_store.search.return_value = [{"payload": {"text": "context chunk"}}]
        mock_gateway.chat_completion.return_value = {
            "choices": [{"message": {"content": "an answer"}}],
            "usage": {"total_tokens": 100},
            "response_cost_usd": 0.002,
        }
        mock_judge.return_value = {"safety": 9, "correctness": 9, "relevance": 9}
        mock_gate.side_effect = [{"passed": True}, {"passed": False}]
        response = rag_evaluate(request)

    assert response.pass_rate == 0.5
    assert response.passed is False
    assert response.total_tokens == 200  # 100 per eval_case, 2 eval_cases
    assert response.total_cost_usd == pytest.approx(0.004)
    mock_gateway.chat_completion.assert_any_call(
        model="llama-3-8b-self-hosted",
        messages=[
            {"role": "system", "content": "Answer using only this context:\n\ncontext chunk"},
            {"role": "user", "content": "q1"},
        ],
    )
    # The judge must see the same retrieved context the answer was generated
    # from — otherwise it can't score faithfulness (see llm_judge.py).
    mock_judge.assert_any_call("q1", "an answer", "context chunk")
    # Retrieval is scoped to the evaluated version, not the whole collection.
    mock_vector_store.search.assert_any_call(
        [0.1], top_k=5, collection="smoke-test", index_version="1"
    )


def test_rag_evaluate_reports_none_cost_when_model_has_no_pricing() -> None:
    # No cost entry -> LiteLLM omits the header -> response_cost_usd is None.
    request = RagEvaluateRequest(
        collection="smoke-test",
        index_version="1",
        eval_cases=[RagEvalCase(question="q1")],
    )
    with (
        patch("routers.rag.llm_gateway_adapter") as mock_gateway,
        patch("routers.rag.vector_store_adapter") as mock_vector_store,
        patch("routers.rag.judge_response") as mock_judge,
        patch("routers.rag.evaluate_gate") as mock_gate,
    ):
        mock_gateway.embed.return_value = [[0.1]]
        mock_vector_store.search.return_value = [{"payload": {"text": "context chunk"}}]
        mock_gateway.chat_completion.return_value = {
            "choices": [{"message": {"content": "an answer"}}],
            "usage": {"total_tokens": 50},
            "response_cost_usd": None,
        }
        mock_judge.return_value = {"safety": 9, "correctness": 9, "relevance": 9}
        mock_gate.return_value = {"passed": True}
        response = rag_evaluate(request)

    assert response.total_tokens == 50
    assert response.total_cost_usd is None


def test_list_rag_collections_returns_names_from_registry() -> None:
    with patch("routers.rag.registry_adapter") as mock_registry:
        mock_registry.list_names.return_value = ["idp-docs", "smoke-test"]
        response = list_rag_collections()

    mock_registry.list_names.assert_called_once_with("rag-index")
    assert response.names == ["idp-docs", "smoke-test"]


def test_list_rag_sources_returns_repo_relative_docs() -> None:
    response = list_rag_sources()

    assert "docs/architecture-overview.md" in response.sources
    assert all(source.startswith("docs/") for source in response.sources)


def test_list_rag_collection_versions_returns_sorted_versions() -> None:
    with patch("routers.rag.registry_adapter") as mock_registry:
        mock_registry.list_versions.return_value = {"2": {}, "1": {}, "10": {}}
        response = list_rag_collection_versions("smoke-test")

    mock_registry.list_versions.assert_called_once_with("rag-index", "smoke-test")
    assert response.versions == ["1", "2", "10"]


def test_rag_activate_calls_set_active_version() -> None:
    request = RagActivateRequest(collection="smoke-test", index_version="1")
    with (
        patch("routers.rag.registry_adapter") as mock_registry,
        patch("routers.rag.eval_result_adapter") as mock_eval_result,
        patch("routers.rag.DEPLOYMENT_EVENTS") as mock_deployment_events,
        patch("routers.rag.deployment_event_store"),
    ):
        mock_eval_result.get_last_failure_at.return_value = None
        response = rag_activate(request)

    mock_registry.set_active_version.assert_called_once_with(
        "rag-index", "smoke-test", "1", "production"
    )
    assert response.active_version == "1"
    mock_deployment_events.labels.assert_called_once_with(
        track="llmops",
        subject_type="rag-index",
        subject_id="smoke-test",
        event_type="deploy",
    )


def test_rag_activate_threads_environment_to_registry_and_event_store() -> None:
    request = RagActivateRequest(collection="smoke-test", index_version="1", environment="staging")
    with (
        patch("routers.rag.registry_adapter") as mock_registry,
        patch("routers.rag.eval_result_adapter") as mock_eval_result,
        patch("routers.rag.DEPLOYMENT_EVENTS"),
        patch("routers.rag.deployment_event_store") as mock_event_store,
    ):
        mock_eval_result.get_last_failure_at.return_value = None
        response = rag_activate(request)

    assert response.environment == "staging"
    mock_registry.set_active_version.assert_called_once_with(
        "rag-index", "smoke-test", "1", "staging"
    )
    assert mock_event_store.record_event.call_args.kwargs["environment_name"] == "staging"


def test_rag_activate_tags_rollback_event_type() -> None:
    request = RagActivateRequest(collection="smoke-test", index_version="1", is_rollback=True)
    with (
        patch("routers.rag.registry_adapter"),
        patch("routers.rag.eval_result_adapter") as mock_eval_result,
        patch("routers.rag.DEPLOYMENT_EVENTS") as mock_deployment_events,
        patch("routers.rag.deployment_event_store") as mock_event_store,
    ):
        mock_eval_result.get_last_failure_at.return_value = None
        rag_activate(request)

    mock_deployment_events.labels.assert_called_once_with(
        track="llmops",
        subject_type="rag-index",
        subject_id="smoke-test",
        event_type="rollback",
    )
    assert mock_event_store.record_event.call_args.kwargs["name"].startswith(
        "rag-rollback-smoke-test-"
    )
