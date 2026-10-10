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
from session_store.base import (Experiment, ExperimentRun, Message,
                                MessageObservability, RunResult, Session)

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
  -- 可观测性字段（M5）：token 计数、参数快照、渲染 prompt、分段统计与耗时
  input_tokens INTEGER, output_tokens INTEGER, params_snapshot TEXT,
  prompt_text TEXT, prompt_messages TEXT, prompt_seg_tokens TEXT,
  prompt_eval_tokens INTEGER,
  ttft_ms REAL, prefill_ms REAL, decode_ms REAL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL  -- JSON 字符串
);

-- M6 参数实验场：与正式会话完全隔离，观测列口径同 messages 表
CREATE TABLE IF NOT EXISTS experiments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  prompt TEXT NOT NULL,
  system_prompt TEXT NOT NULL DEFAULT '',  -- 创建时的全局 system prompt 快照
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS experiment_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
  seq INTEGER NOT NULL,                   -- 栏位顺序 0/1/2
  params TEXT NOT NULL,                   -- JSON {temperature, top_p, top_k, max_tokens}
  status TEXT NOT NULL DEFAULT 'pending', -- pending/running/done/stopped/error
  content TEXT NOT NULL DEFAULT '',
  error TEXT,                             -- 失败时的 {code, message} JSON
  created_at REAL NOT NULL,
  finished_at REAL,
  input_tokens INTEGER, output_tokens INTEGER, prompt_eval_tokens INTEGER,
  prompt_text TEXT, prompt_messages TEXT, prompt_seg_tokens TEXT,
  ttft_ms REAL, prefill_ms REAL, decode_ms REAL
);

CREATE INDEX IF NOT EXISTS idx_runs_experiment
  ON experiment_runs(experiment_id, seq);
"""

# user_version 1：M5 观测列；2：M6 experiments/experiment_runs 两表
# 老库（一期建成，version 0）走 ALTER TABLE 补列；
# M6 两表在建库脚本里就是 IF NOT EXISTS，新老库都覆盖，迁移只推版本号
_SCHEMA_VERSION = 2
_M5_COLUMNS = [
    "prompt_text TEXT",
    "prompt_messages TEXT",
    "prompt_seg_tokens TEXT",
    "prompt_eval_tokens INTEGER",
    "ttft_ms REAL",
    "prefill_ms REAL",
    "decode_ms REAL",
]

_MESSAGE_COLS = ("id, session_id, role, content, created_at,"
                 " input_tokens, output_tokens, params_snapshot,"
                 " prompt_text, prompt_messages, prompt_seg_tokens,"
                 " prompt_eval_tokens, ttft_ms, prefill_ms, decode_ms")

_EXPERIMENT_COLS = "id, prompt, system_prompt, created_at"

_RUN_COLS = ("id, experiment_id, seq, params, status, content, error,"
             " created_at, finished_at,"
             " input_tokens, output_tokens, prompt_eval_tokens,"
             " prompt_text, prompt_messages, prompt_seg_tokens,"
             " ttft_ms, prefill_ms, decode_ms")


class SqliteSessionStore:
    """方法签名与语义见 base.py 的接口约定。"""

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")  # 级联删除的前提
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._migrate()

    def _migrate(self):
        """按 PRAGMA user_version 补列；新库（schema 已含全部列）只推进版本号。"""
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version >= _SCHEMA_VERSION:
            return
        if version < 1:
            existing = {row[1] for row in
                        self._conn.execute("PRAGMA table_info(messages)")}
            for column in _M5_COLUMNS:
                if column.split()[0] not in existing:
                    self._conn.execute(
                        f"ALTER TABLE messages ADD COLUMN {column}")
        self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        self._conn.commit()

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
                       obs: MessageObservability):
        """回填生成完毕（含被停止）的 assistant 消息内容与观测数据。"""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE messages SET content = ?, input_tokens = ?,"
                " output_tokens = ?, params_snapshot = ?,"
                " prompt_text = ?, prompt_messages = ?,"
                " prompt_seg_tokens = ?, prompt_eval_tokens = ?,"
                " ttft_ms = ?, prefill_ms = ?, decode_ms = ?"
                " WHERE id = ?",
                (content, obs.input_tokens, obs.output_tokens,
                 obs.params_snapshot, obs.prompt_text, obs.prompt_messages,
                 obs.prompt_seg_tokens, obs.prompt_eval_tokens,
                 obs.ttft_ms, obs.prefill_ms, obs.decode_ms, message_id))
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ("
                "SELECT session_id FROM messages WHERE id = ?)",
                (now, message_id))
            self._conn.commit()

    def get_message(self, message_id: int) -> Message:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_MESSAGE_COLS} FROM messages WHERE id = ?",
                (message_id,)).fetchone()
        if row is None:
            raise ServiceError(
                "MESSAGE_NOT_FOUND", f"消息 {message_id} 不存在")
        return Message(*row)

    def get_messages(self, session_id: int) -> list[Message]:
        self.get_session(session_id)  # 不存在抛 SESSION_NOT_FOUND
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_MESSAGE_COLS}"
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

    # ---------- 实验（M6 参数实验场） ----------

    def create_experiment(self, prompt: str, system_prompt: str,
                          param_sets: list[str]) -> Experiment:
        """建实验并连 runs 一起插入（seq 按 param_sets 顺序，status=pending）。

        param_sets 元素为已校验过的四参数 JSON 字符串。
        """
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO experiments (prompt, system_prompt, created_at)"
                " VALUES (?, ?, ?)", (prompt, system_prompt, now))
            experiment_id = cur.lastrowid
            self._conn.executemany(
                "INSERT INTO experiment_runs"
                " (experiment_id, seq, params, created_at)"
                " VALUES (?, ?, ?, ?)",
                [(experiment_id, seq, params, now)
                 for seq, params in enumerate(param_sets)])
            self._conn.commit()
            return Experiment(id=experiment_id, prompt=prompt,
                              system_prompt=system_prompt, created_at=now)

    def get_experiment(self, experiment_id: int) -> Experiment:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_EXPERIMENT_COLS} FROM experiments WHERE id = ?",
                (experiment_id,)).fetchone()
        if row is None:
            raise ServiceError("EXPERIMENT_NOT_FOUND",
                               f"实验 {experiment_id} 不存在")
        return Experiment(*row)

    def list_experiments(self) -> list[Experiment]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_EXPERIMENT_COLS} FROM experiments"
                " ORDER BY id DESC").fetchall()
        return [Experiment(*r) for r in rows]

    def delete_experiment(self, experiment_id: int):
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM experiments WHERE id = ?", (experiment_id,))
            self._conn.commit()
        if cur.rowcount == 0:
            raise ServiceError("EXPERIMENT_NOT_FOUND",
                               f"实验 {experiment_id} 不存在")

    def get_runs(self, experiment_id: int) -> list[ExperimentRun]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_RUN_COLS} FROM experiment_runs"
                " WHERE experiment_id = ? ORDER BY seq",
                (experiment_id,)).fetchall()
        return [ExperimentRun(*r) for r in rows]

    def get_run(self, experiment_id: int, run_id: int) -> ExperimentRun:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_RUN_COLS} FROM experiment_runs WHERE id = ?",
                (run_id,)).fetchone()
        if row is None or row[1] != experiment_id:
            raise ServiceError(
                "RUN_NOT_FOUND",
                f"实验 {experiment_id} 中不存在 run {run_id}")
        return ExperimentRun(*row)

    def start_run(self, run_id: int):
        """置 running 并清空上轮 error（重试场景）；内容在 finish_run 覆盖。"""
        with self._lock:
            self._conn.execute(
                "UPDATE experiment_runs SET status = 'running', error = NULL"
                " WHERE id = ?", (run_id,))
            self._conn.commit()

    def finish_run(self, run_id: int, result: RunResult):
        """回填生成完毕（含被停止/失败）的 run：终态、内容、观测字段。"""
        with self._lock:
            self._conn.execute(
                "UPDATE experiment_runs SET status = ?, content = ?,"
                " error = ?, finished_at = ?,"
                " input_tokens = ?, output_tokens = ?,"
                " prompt_eval_tokens = ?, prompt_text = ?,"
                " prompt_messages = ?, prompt_seg_tokens = ?,"
                " ttft_ms = ?, prefill_ms = ?, decode_ms = ?"
                " WHERE id = ?",
                (result.status, result.content, result.error, time.time(),
                 result.input_tokens, result.output_tokens,
                 result.prompt_eval_tokens, result.prompt_text,
                 result.prompt_messages, result.prompt_seg_tokens,
                 result.ttft_ms, result.prefill_ms, result.decode_ms,
                 run_id))
            self._conn.commit()


session_store = SqliteSessionStore(config.db_path)
