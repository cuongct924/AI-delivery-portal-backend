"""adapters/ai_platform/chat_session_store.py — real SQLite in a tmp_path,
no mocking (same spirit as the prediction-log adapter's tests)."""

from adapters.ai_platform.chat_session_store import SqliteChatSessionStore


def _store(tmp_path) -> SqliteChatSessionStore:
    return SqliteChatSessionStore(path=str(tmp_path / "sessions.db"))


def test_get_returns_none_for_unknown_session(tmp_path) -> None:
    assert _store(tmp_path).get("nope") is None


def test_append_messages_creates_then_extends(tmp_path) -> None:
    store = _store(tmp_path)
    store.append_messages("s1", "dev", [{"role": "user", "content": "hi"}])
    store.append_messages("s1", "dev", [{"role": "assistant", "content": "hello"}])

    session = store.get("s1")
    assert session is not None
    assert session["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_merge_draft_accumulates_across_calls(tmp_path) -> None:
    store = _store(tmp_path)
    store.merge_draft("s1", "dev", "train-track-register", {"modelName": "fraud"})
    merged = store.merge_draft("s1", "dev", "train-track-register", {"gpuType": "A100"})

    assert merged["form_data"] == {"modelName": "fraud", "gpuType": "A100"}


def test_merge_draft_resets_when_template_changes(tmp_path) -> None:
    store = _store(tmp_path)
    store.merge_draft("s1", "dev", "train-track-register", {"modelName": "fraud"})
    merged = store.merge_draft("s1", "dev", "llm-serve-deploy", {"gpuType": "A100"})

    assert merged["form_data"] == {"gpuType": "A100"}


def test_clear_removes_session(tmp_path) -> None:
    store = _store(tmp_path)
    store.append_messages("s1", "dev", [{"role": "user", "content": "hi"}])
    store.clear("s1")
    assert store.get("s1") is None
