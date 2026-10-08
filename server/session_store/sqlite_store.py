"""SQLite 实现的会话存储（表结构见技术方案 §4）。

单连接 + 锁：推理在工作线程，连接须 check_same_thread=False；
单用户场景下串行化访问足够，也是线程安全的底线。
"""
import sqlite3
import threading
import time
from pathlib import Path

from config import config
from model_manager.manager import ServiceError
from session_store.base import Message, Session

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL DEFAULT '新会话',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
  content TEXT NOT NULL,
  created_at REAL NOT NULL,
  -- 可观测性字段：token 计数随消息落库；params_snapshot 一期恒为 NULL（PRD 3.7）
  input_tokens INTEGER, output_tokens INTEGER, params_snapshot TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL  -- JSON 字符串
);
"""


class SqliteSessionStore:
    """方法签名与语义见 base.py 的接口约定。"""

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")  # 级联删除的前提
        with self._lock:
            self._conn.executescript(_SCHEMA)

    # ---------- 会话 ----------

    def create_session(self, title: str) -> Session:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions (title, created_at, updated_at)"
                " VALUES (?, ?, ?)", (title, now, now))
            self._conn.commit()
            return Session(id=cur.lastrowid, title=title,
                           created_at=now, updated_at=now)

    def get_session(self, session_id: int) -> Session:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, title, created_at, updated_at"
                " FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise ServiceError("SESSION_NOT_FOUND", f"会话 {session_id} 不存在")
        return Session(*row)

    def list_sessions(self) -> list[Session]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, title, created_at, updated_at"
                " FROM sessions ORDER BY updated_at DESC").fetchall()
        return [Session(*r) for r in rows]

    def rename_session(self, session_id: int, title: str) -> Session:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (title, session_id))
            self._conn.commit()
        if cur.rowcount == 0:
            raise ServiceError("SESSION_NOT_FOUND", f"会话 {session_id} 不存在")
        return self.get_session(session_id)

    def delete_session(self, session_id: int):
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,))
            self._conn.commit()
        if cur.rowcount == 0:
            raise ServiceError("SESSION_NOT_FOUND", f"会话 {session_id} 不存在")

    # ---------- 消息 ----------

    def add_message(self, session_id: int, role: str, content: str) -> Message:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at)"
                " VALUES (?, ?, ?, ?)", (session_id, role, content, now))
            # 会话活跃时间随消息推进，列表按它倒序
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id))
            self._conn.commit()
            return Message(id=cur.lastrowid, session_id=session_id,
                           role=role, content=content, created_at=now)

    def update_message(self, message_id: int, content: str,
                       input_tokens: int | None = None,
                       output_tokens: int | None = None):
        """回填生成完毕（含被停止）的 assistant 消息内容与 token 计数。"""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE messages SET content = ?, input_tokens = ?,"
                " output_tokens = ? WHERE id = ?",
                (content, input_tokens, output_tokens, message_id))
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ("
                "SELECT session_id FROM messages WHERE id = ?)",
                (now, message_id))
            self._conn.commit()

    def get_messages(self, session_id: int) -> list[Message]:
        self.get_session(session_id)  # 不存在抛 SESSION_NOT_FOUND
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, session_id, role, content, created_at,"
                " input_tokens, output_tokens, params_snapshot"
                " FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,)).fetchall()
        return [Message(*r) for r in rows]

    # ---------- 设置（settings 表，供 params 模块用） ----------

    def get_setting(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_setting(self, key: str, value: str):
        with self._lock:
            self._conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value))
            self._conn.commit()


session_store = SqliteSessionStore(config.db_path)
