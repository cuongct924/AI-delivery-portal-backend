"""SQLite-backed IChatSessionStore — same "local file, no server to run"
precedent as adapters/ai_platform/prediction_log_adapter.py. Holds each
chat session's message history and current template draft so a browser
reload restores both (the frontend keeps only the opaque `session_id`).

TTL-evicted on write: a session untouched for `ttl_hours` is dropped, so
the table can't grow unbounded. Swapping to Postgres later is one new
class implementing IChatSessionStore.
"""

import json
import os
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from adapters.ai_platform.interfaces import ChatDraft, ChatSession, IChatSessionStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    user_ref TEXT NOT NULL,
    messages_json TEXT NOT NULL,
    draft_json TEXT,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at ON chat_sessions (updated_at);
"""

DEFAULT_TTL_HOURS = 24


class SqliteChatSessionStore(IChatSessionStore):
    def __init__(self, path: str | None = None, ttl_hours: int = DEFAULT_TTL_HOURS):
        self.path = Path(path or os.getenv("CHAT_SESSION_PATH", ".state/chat_sessions.db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def get(self, session_id: str) -> ChatSession | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id, user_ref, messages_json, draft_json, updated_at "
                "FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row[0],
            "user_ref": row[1],
            "messages": json.loads(row[2]),
            "draft": json.loads(row[3]) if row[3] is not None else None,
            "updated_at": row[4],
        }

    def append_messages(
        self, session_id: str, user_ref: str, messages: Sequence[Mapping[str, str]]
    ) -> None:
        with self._lock, self._connect() as conn:
            self._evict(conn)
            existing = conn.execute(
                "SELECT messages_json FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            history = json.loads(existing[0]) if existing is not None else []
            history.extend(messages)
            conn.execute(
                "INSERT INTO chat_sessions "
                "(session_id, user_ref, messages_json, draft_json, updated_at) "
                "VALUES (?, ?, ?, NULL, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "messages_json = excluded.messages_json, updated_at = excluded.updated_at",
                (session_id, user_ref, json.dumps(history), _now()),
            )

    def merge_draft(
        self, session_id: str, user_ref: str, template: str, patch: Mapping[str, object]
    ) -> ChatDraft:
        with self._lock, self._connect() as conn:
            self._evict(conn)
            existing = conn.execute(
                "SELECT draft_json FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            current: dict[str, object] = {}
            if existing is not None and existing[0] is not None:
                stored = json.loads(existing[0])
                # A different template starts a fresh draft — merging fields
                # across templates would leave stale keys behind.
                if stored.get("template") == template:
                    current = stored.get("form_data", {})
            merged: ChatDraft = {
                "template": template,
                "form_data": {**current, **patch},
                "updated_at": _now(),
            }
            conn.execute(
                "INSERT INTO chat_sessions "
                "(session_id, user_ref, messages_json, draft_json, updated_at) "
                "VALUES (?, ?, '[]', ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "draft_json = excluded.draft_json, updated_at = excluded.updated_at",
                (session_id, user_ref, json.dumps(merged), _now()),
            )
        return merged

    def clear(self, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))

    def _evict(self, conn: sqlite3.Connection) -> None:
        cutoff = (datetime.now(UTC) - self.ttl).isoformat()
        conn.execute("DELETE FROM chat_sessions WHERE updated_at < ?", (cutoff,))

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)


def _now() -> str:
    return datetime.now(UTC).isoformat()
