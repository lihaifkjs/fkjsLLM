// 接口契约类型，严格对齐 docs/技术方案.md §3

// ---------- 对话（§3.1） ----------

export interface ChatRequest {
  session_id: number | null
  content: string
}

export interface ChatMeta {
  session_id: number
  user_message_id: number
  assistant_message_id: number
  title: string
}

export interface ChatToken {
  text: string
}

export interface Usage {
  input_tokens: number
  output_tokens: number
}

// M5：done 帧新增分段计时（服务端 llama_perf_context 真值）
export interface Timing {
  ttft_ms: number
  prefill_ms: number
  decode_ms: number
}

export interface ChatDone {
  usage: Usage
  timing: Timing
}

export interface ApiError {
  code: string
  message: string
}

export type ChatSseEvent = 'meta' | 'token' | 'tool_call' | 'done' | 'error'

// M9：工具调用中间态事件（开始搜索时下发；回答轮首个 token 到达即结束）
export interface ToolCallEvent {
  name: string
  query: string
}

// 一帧原始 SSE 记录：time 为到达时刻（秒），data 为「data: 」后的原始字符串（不重新序列化，保真）
export interface SseFrame {
  time: number
  event: string
  data: string
}

export interface ChatStreamCallbacks {
  onMeta: (meta: ChatMeta) => void
  onToken: (token: ChatToken) => void
  onDone: (done: ChatDone) => void
  onError: (error: ApiError) => void
  // M9：工具调用中间态（可选，mock 不触发）
  onToolCall?: (toolCall: ToolCallEvent) => void
  // 可选：每一帧的原始记录，用于调试抽屉「原始输出」
  onFrame?: (frame: SseFrame) => void
}

// 聊天流的本地句柄：abort() 只终止本地读取，服务端中断走 stopChat
export interface ChatStreamHandle {
  abort: () => void
}

// ---------- 会话（§3.2） ----------

export interface SessionSummary {
  id: number
  title: string
  updated_at: number
  // M7：会话级知识库检索开关
  rag_enabled: boolean
  // M9：会话级网页搜索工具开关
  tool_enabled: boolean
}

export interface Session extends SessionSummary {
  created_at: number
}

export type MessageRole = 'user' | 'assistant' | 'system'

export interface ChatMessage {
  id: number
  role: MessageRole
  content: string
  created_at: number
}

// GET /sessions/{sid}/messages 出参（M6.5 起包一层）：
// summary_covered_upto 为最新压缩记录覆盖到的最大消息 id（细节已丢失）；
// 存在压缩记录即返回（与当前策略无关），无摘要时为 null
export interface SessionMessages {
  messages: ChatMessage[]
  summary_covered_upto: number | null
}

// ---------- 参数（§3.3） ----------

export interface InferenceParams {
  temperature: number
  top_p: number
  top_k: number
  repetition_penalty: number
  max_tokens: number
  system_prompt: string
  // M6.5：上下文超限策略（全局开关），滑窗硬截断 / 摘要压缩
  context_strategy: 'sliding_window' | 'summary'
  preset: string
}

// ---------- 模型开关（§3.4） ----------

export type ModelState = 'unloaded' | 'loading' | 'loaded' | 'unloading' | 'error'

export interface ModelStatus {
  state: ModelState
  vram_used_mb: number | null
  error: string | null
  // 仅 GET /api/model/status 返回（load/unload 响应不含），前端算上下文预算用
  n_ctx?: number
  // M8：当前选定模型条目与可选列表（多模型注册表）
  model_id?: string
  available_models?: ModelInfo[]
}

export interface ModelInfo {
  id: string
  name: string
}

// ---------- 观测数据（M5，docs/M5开发计划.md §4.2） ----------

// prompt 分段 token 口径：system / 历史 / 当前提问，三段之和 = input_tokens
export interface SegTokens {
  system: number
  history: number
  current: number
}

// 生成时生效的参数快照；source 标注来源（stored=用户已保存设置 / default=全局默认值）
export interface ParamsSnapshot extends InferenceParams {
  source: 'stored' | 'default'
}

export interface PromptMessage {
  role: MessageRole
  content: string
}

// ---------- 摘要压缩（M6.5，docs/M6.5开发计划.md §3/§6.5） ----------
// 本消息触发的那次压缩：完整记录（压缩前 prompt / 摘要 / token 对比 / 耗时）
export interface CompressionRecord {
  triggered: true
  id: number
  prev_summary: string
  source_messages: PromptMessage[]
  source_prompt: string
  summary: string
  covered_upto: number
  source_tokens: number | null
  summary_tokens: number | null
  input_tokens: number | null
  output_tokens: number | null
  timing: Timing
}

// 本轮未触发新压缩，复用既有摘要
export interface CompressionReuse {
  triggered: false
  active_id: number
}

export type CompressionInfo = CompressionRecord | CompressionReuse

// ---------- RAG 知识库（M7，docs/M7开发计划.md §3/§8） ----------

export interface KnowledgeDocument {
  id: number
  filename: string
  char_count: number
  chunk_count: number
  created_at: number
}

// 命中片段快照（retrievals.hits JSON）：与文档生命周期解耦，文档删除后仍可读
export interface RetrievalHit {
  chunk_id: number
  document_id: number
  filename: string
  score: number
  content: string
}

// 一轮检索的调用链记录；hits 可空数组（无命中）
export interface Retrieval {
  query: string
  hits: RetrievalHit[]
  injected_tokens: number | null
  latency_ms: number | null
}

// ---------- 网页搜索工具（M9，docs/M9开发计划.md §3） ----------

// 搜索结果快照条目（tool_calls.results JSON）；content 为 trafilatura 正文（未抓时为 null）
export interface ToolCallResult {
  title: string
  url: string
  snippet: string
  content: string | null
}

// 一轮工具调用的调用链记录；status: success / search_error / parse_error
// arguments 为解析后的对象（parse_error 时为原始文本）
export interface ToolCallRecord {
  name: string
  arguments: Record<string, unknown> | string
  query: string
  status: 'success' | 'search_error' | 'parse_error'
  results: ToolCallResult[]
  inject_tokens: number | null
  latency_ms: number | null
  error: string | null
}

// GET /sessions/{sid}/messages/{mid}/observability 出参；
// user 消息或 M5 前的老消息各字段为 null，pane 据此降级到帧统计
export interface Observability {
  input_tokens: number | null
  output_tokens: number | null
  // 本轮实际 prefill 算力（n_p_eval），KV 前缀复用时会小于 input_tokens
  prompt_eval_tokens: number | null
  seg_tokens: SegTokens | null
  timing: Timing | null
  params_snapshot: ParamsSnapshot | null
  prompt_text: string | null
  prompt_messages: PromptMessage[] | null
  // M6.5：压缩调用记录（触发/复用/无 三态）
  compression: CompressionInfo | null
  // M7：知识库检索记录（会话启用检索且有落库 → 完整记录含无命中 hits:[]；
  // 未启用 / 开关后开的老消息 → null）
  retrieval: Retrieval | null
  // M9：工具调用记录（会话启用工具且本轮触发 → 完整记录含失败态；否则 null）
  tool_call: ToolCallRecord | null
}

// ---------- 统一 API 形状 ----------

// api/index.ts 按 VITE_API_MODE 导出 mock 或 client，二者都满足该接口
export interface LlmApi {
  chat: (req: ChatRequest, cb: ChatStreamCallbacks) => ChatStreamHandle
  stopChat: (sessionId: number) => Promise<void>
  listSessions: () => Promise<SessionSummary[]>
  createSession: (title?: string) => Promise<Session>
  getSessionMessages: (sessionId: number) => Promise<SessionMessages>
  renameSession: (sessionId: number, title: string) => Promise<Session>
  deleteSession: (sessionId: number) => Promise<void>
  getObservability: (sessionId: number, messageId: number) => Promise<Observability>
  // 手动压缩全部历史（批次4）：返回本次压缩记录（triggered: true 完整形状）
  compressSession: (sessionId: number) => Promise<CompressionRecord>
  // M7：会话级知识库检索开关（PATCH 部分更新）
  setRagEnabled: (sessionId: number, enabled: boolean) => Promise<Session>
  // M9：会话级网页搜索工具开关（PATCH 部分更新）
  setToolEnabled: (sessionId: number, enabled: boolean) => Promise<Session>
  // M7：知识库文档管理
  listDocuments: () => Promise<KnowledgeDocument[]>
  uploadDocument: (file: File) => Promise<KnowledgeDocument>
  deleteDocument: (documentId: number) => Promise<void>
  getParams: () => Promise<InferenceParams>
  // 部分字段补丁（技术方案 §3.3）：显式字段优先于 preset；preset 仅在精确/平衡/创意三选一
  // 时传入，勿传 'custom'（后端视为未知预设，400 INVALID_PARAMS）
  saveParams: (patch: Partial<InferenceParams>) => Promise<InferenceParams>
  getModelStatus: () => Promise<ModelStatus>
  loadModel: (modelId?: string) => Promise<ModelStatus>
  unloadModel: () => Promise<ModelStatus>
}
