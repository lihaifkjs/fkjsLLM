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

export interface ChatDone {
  usage: Usage
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

// ---------- 参数（§3.3） ----------

export interface InferenceParams {
  temperature: number
  top_p: number
  top_k: number
  repetition_penalty: number
  max_tokens: number
  system_prompt: string
  preset: string
}

// ---------- 模型开关（§3.4） ----------

export type ModelState = 'unloaded' | 'loading' | 'loaded' | 'unloading' | 'error'

export interface ModelStatus {
  state: ModelState
  vram_used_mb: number | null
  error: string | null
}

// ---------- 统一 API 形状 ----------

// api/index.ts 按 VITE_API_MODE 导出 mock 或 client，二者都满足该接口
export interface LlmApi {
  chat: (req: ChatRequest, cb: ChatStreamCallbacks) => ChatStreamHandle
  stopChat: (sessionId: number) => Promise<void>
  listSessions: () => Promise<SessionSummary[]>
  createSession: (title?: string) => Promise<Session>
  getSessionMessages: (sessionId: number) => Promise<ChatMessage[]>
  getParams: () => Promise<InferenceParams>
  saveParams: (params: InferenceParams) => Promise<InferenceParams>
  getModelStatus: () => Promise<ModelStatus>
  loadModel: () => Promise<ModelStatus>
  unloadModel: () => Promise<ModelStatus>
}
