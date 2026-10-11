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

export type ChatSseEvent = 'meta' | 'token' | 'done' | 'error'

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
// summary_covered_upto 为当前生效摘要覆盖到的最大消息 id（细节已丢失）；
// 滑窗策略或无摘要时为 null
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
  getParams: () => Promise<InferenceParams>
  // 部分字段补丁（技术方案 §3.3）：显式字段优先于 preset；preset 仅在精确/平衡/创意三选一
  // 时传入，勿传 'custom'（后端视为未知预设，400 INVALID_PARAMS）
  saveParams: (patch: Partial<InferenceParams>) => Promise<InferenceParams>
  getModelStatus: () => Promise<ModelStatus>
  loadModel: () => Promise<ModelStatus>
  unloadModel: () => Promise<ModelStatus>
}
