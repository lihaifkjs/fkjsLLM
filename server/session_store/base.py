"""会话存储接口：数据结构与类型化签名约定（技术方案 1.1）。

按已定决策不用 abc.ABC 强制继承——实现类（sqlite_store.SqliteSessionStore）
结构对齐本文件的签名即可，换库只换实现。
接口方法清单：
  create_session(title) -> Session
  get_session(session_id) -> Session          # 不存在抛 SESSION_NOT_FOUND
  list_sessions() -> list[Session]            # 按 updated_at 倒序
  rename_session(session_id, title) -> Session
  set_rag_enabled(session_id, enabled) -> Session  # M7 会话级检索开关
  delete_session(session_id)                  # 级联删消息
  add_message(session_id, role, content) -> Message
  update_message(message_id, content, obs: MessageObservability)
  get_messages(session_id) -> list[Message]   # 按 id 升序
  get_message(message_id) -> Message          # 不存在抛 MESSAGE_NOT_FOUND
  get_setting(key) -> str | None              # value 为 JSON 字符串
  set_setting(key, value)
  # M6.5 摘要压缩（compressions 表，兼任压缩历史与会话当前生效摘要缓存）
  add_compression(record: CompressionInput) -> Compression
  get_latest_compression(session_id) -> Compression | None  # 无记录返回 None
  get_compression(compression_id) -> Compression
                                            # 不存在抛 COMPRESSION_NOT_FOUND
  get_compression_by_trigger(session_id, trigger_message_id)
      -> Compression | None                 # 该消息是否触发过压缩（调用链用）
  # M7 RAG 知识库（documents/chunks/retrievals 三表）
  add_document(filename, content, char_count, chunk_count) -> Document
  get_document(document_id) -> Document     # 不存在抛 DOCUMENT_NOT_FOUND
  list_documents() -> list[Document]        # 按 id 倒序（最新上传在前）
  delete_document(document_id)              # 级联删 chunks；不动 retrievals 快照
  add_chunks(document_id, items)            # items: [(seq, content, embedding_blob)]
  get_all_chunks() -> list[Chunk]           # 检索全表扫描用
  add_retrieval(record: RetrievalInput) -> Retrieval
  get_retrieval_by_trigger(session_id, trigger_message_id)
      -> Retrieval | None                   # 该消息触发的那次检索（调用链用）
"""
from dataclasses import dataclass


@dataclass
class Session:
    id: int
    title: str
    created_at: float
    updated_at: float
    rag_enabled: int = 0  # M7：会话级知识库检索开关（0/1）


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


@dataclass
class CompressionInput:
    """一次摘要压缩的入库载荷（M6.5）：压缩输入 + 产出 + 调用观测。

    prev_summary 为增量追加的既有摘要（首次为空串）；
    source_messages 为本次新纳入压缩的消息 JSON 字符串 [{role, content}]；
    source_prompt 为压缩调用的完整渲染 prompt 文本。
    """
    session_id: int
    trigger_message_id: int  # 触发压缩的 assistant 占位消息 id；0 = 手动压缩（无对应消息）
    prev_summary: str
    source_messages: str
    source_prompt: str
    summary: str
    covered_upto: int  # 摘要覆盖到的最大 message id（增量游标，单调不减）
    source_tokens: int | None = None
    summary_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_ms: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None


@dataclass
class Compression:
    """一条压缩记录（M6.5）；会话当前生效摘要 = 该 session id 最大的一条。"""
    id: int
    session_id: int
    trigger_message_id: int
    prev_summary: str
    source_messages: str
    source_prompt: str
    summary: str
    covered_upto: int
    source_tokens: int | None
    summary_tokens: int | None
    input_tokens: int | None
    output_tokens: int | None
    ttft_ms: float | None
    prefill_ms: float | None
    decode_ms: float | None
    created_at: float


@dataclass
class Document:
    """知识库文档（M7）：原文整体落库，全局文档池、不属于任何会话。"""
    id: int
    filename: str
    content: str
    char_count: int
    chunk_count: int
    created_at: float


@dataclass
class Chunk:
    """文档切块（M7）：embedding 为 float32 小端字节流（dim=512）。"""
    id: int
    document_id: int
    seq: int  # 文档内顺序
    content: str
    char_count: int
    embedding: bytes
    created_at: float


@dataclass
class RetrievalInput:
    """一次检索的入库载荷（M7）。

    hits 为命中片段快照 JSON 字符串 [{chunk_id, document_id, filename,
    score, content}]——不回联 chunks，文档删除后历史调用链仍可读。
    """
    session_id: int
    trigger_message_id: int  # 本轮 assistant 占位消息 id
    query: str
    hits: str
    injected_tokens: int | None = None
    latency_ms: float | None = None


@dataclass
class Retrieval:
    id: int
    session_id: int
    trigger_message_id: int
    query: str
    hits: str
    injected_tokens: int | None
    latency_ms: float | None
    created_at: float
