"""会话存储接口：数据结构与类型化签名约定（技术方案 1.1）。

按已定决策不用 abc.ABC 强制继承——实现类（sqlite_store.SqliteSessionStore）
结构对齐本文件的签名即可，换库只换实现。
接口方法清单：
  create_session(title) -> Session
  get_session(session_id) -> Session          # 不存在抛 SESSION_NOT_FOUND
  list_sessions() -> list[Session]            # 按 updated_at 倒序
  rename_session(session_id, title) -> Session
  delete_session(session_id)                  # 级联删消息
  add_message(session_id, role, content) -> Message
  update_message(message_id, content, obs: MessageObservability)
  get_messages(session_id) -> list[Message]   # 按 id 升序
  get_message(message_id) -> Message          # 不存在抛 MESSAGE_NOT_FOUND
  get_setting(key) -> str | None              # value 为 JSON 字符串
  set_setting(key, value)
  # M6 参数实验场（experiments / experiment_runs 两表，与会话数据隔离）
  create_experiment(prompt, system_prompt, param_sets) -> Experiment
                                            # 连 runs 一起建（status=pending）
  get_experiment(experiment_id) -> Experiment
                                            # 不存在抛 EXPERIMENT_NOT_FOUND
  list_experiments() -> list[Experiment]    # 按 id 倒序（新的在前）
  delete_experiment(experiment_id)          # 级联删 runs
  get_runs(experiment_id) -> list[ExperimentRun]   # 按 seq 升序
  get_run(experiment_id, run_id) -> ExperimentRun  # 不存在/不属于抛 RUN_NOT_FOUND
  start_run(run_id)                         # 置 running 并清空 error
  finish_run(run_id, result: RunResult)     # 回填状态/内容/观测字段/finished_at
"""
from dataclasses import dataclass


@dataclass
class Session:
    id: int
    title: str
    created_at: float
    updated_at: float


@dataclass
class MessageObservability:
    """assistant 消息的可观测性回填（M5）：全部可空，生成失败/中断时部分为空。

    prompt_* 字段为 JSON 字符串（prompt_text 除外，是纯文本）；
    prompt_eval_tokens 是本轮实际 prefill 的 token 数（llama_perf_context
    的 n_p_eval），KV 前缀复用下会小于账单口径 input_tokens。
    """
    input_tokens: int | None = None
    output_tokens: int | None = None
    params_snapshot: str | None = None
    prompt_text: str | None = None
    prompt_messages: str | None = None
    prompt_seg_tokens: str | None = None
    prompt_eval_tokens: int | None = None
    ttft_ms: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None


@dataclass
class Experiment:
    """一次参数对比实验（M6）：prompt + 创建时的全局 system prompt 快照。"""
    id: int
    prompt: str
    system_prompt: str
    created_at: float


@dataclass
class ExperimentRun:
    """实验中的一组参数与生成结果；观测列口径同 messages 表（M5）。

    params 为 JSON 字符串（temperature/top_p/top_k/max_tokens 四项）；
    error 为 {code, message} 的 JSON 字符串，仅 status=error 时有值。
    """
    id: int
    experiment_id: int
    seq: int
    params: str
    status: str  # pending / running / done / stopped / error
    content: str
    error: str | None = None
    created_at: float = 0.0
    finished_at: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    prompt_eval_tokens: int | None = None
    prompt_text: str | None = None
    prompt_messages: str | None = None
    prompt_seg_tokens: str | None = None
    ttft_ms: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None


@dataclass
class RunResult:
    """finish_run 的回填载荷：终态 + 部分内容 + 观测字段（全部可空）。

    对齐 chat 的 MessageObservability 汇合语义——正常/stop/异常三路径
    都经 finish_run 落库，缺省 None 表示该字段未产出。
    """
    status: str  # done / stopped / error
    content: str = ""
    error: str | None = None  # {code, message} JSON 字符串
    input_tokens: int | None = None
    output_tokens: int | None = None
    prompt_eval_tokens: int | None = None
    prompt_text: str | None = None
    prompt_messages: str | None = None
    prompt_seg_tokens: str | None = None
    ttft_ms: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None


@dataclass
class Message:
    id: int
    session_id: int
    role: str  # user / assistant / system
    content: str
    created_at: float
    # 可观测性字段（M5 起随生成落库，user 消息恒为 None）
    input_tokens: int | None = None
    output_tokens: int | None = None
    params_snapshot: str | None = None
    prompt_text: str | None = None
    prompt_messages: str | None = None
    prompt_seg_tokens: str | None = None
    prompt_eval_tokens: int | None = None
    ttft_ms: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None
