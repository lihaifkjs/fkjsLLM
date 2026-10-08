// mock 实现：VITE_API_MODE=mock（默认）时使用，行为对齐后端契约（docs/后端m3实现.md §2）
// 会话/消息/参数用 localStorage 持久化，模拟后端重启不丢；模型状态机仍为内存态
import type {
  ApiError,
  ChatMessage,
  ChatRequest,
  ChatStreamCallbacks,
  ChatStreamHandle,
  InferenceParams,
  LlmApi,
  ModelState,
  ModelStatus,
  Session,
  SessionSummary,
} from './types'

const TOKEN_INTERVAL_MS = 30
const LOAD_MS = 3000
const UNLOAD_MS = 1000
const DEFAULT_TITLE = '新会话'
const TITLE_LEN = 20 // 一期标题取首条消息前 20 字（对齐 server/chat/engine.py）

const LS_SESSIONS = 'fkjsllm.mock.sessions'
const LS_PARAMS = 'fkjsllm.mock.params'

// 预设回复：覆盖 Markdown、代码块，便于验证渲染
const PRESET_REPLY = `这是 mock 回复（后端就绪后切换 \`VITE_API_MODE=real\` 联调）。

支持 **Markdown** 渲染与代码高亮：

\`\`\`python
def hello(name: str) -> str:
    return f"Hello, {name}!"

print(hello("LAN LLM"))
\`\`\`

- 流式输出按 30ms/token 模拟
- 模型加载约 3s、卸载约 1s

> 你的输入是：「{input}」`

// 内置预设与默认值对齐 server/params/presets.py（预设只覆盖采样四参数）
type SamplingValues = Pick<
  InferenceParams,
  'temperature' | 'top_p' | 'top_k' | 'repetition_penalty'
>

const PRESETS: Record<string, SamplingValues> = {
  precise: { temperature: 0.1, top_p: 0.5, top_k: 10, repetition_penalty: 1.05 },
  balanced: { temperature: 0.7, top_p: 0.8, top_k: 20, repetition_penalty: 1.05 },
  creative: { temperature: 1.0, top_p: 0.9, top_k: 40, repetition_penalty: 1.05 },
}

type ParamValues = Omit<InferenceParams, 'preset'>

const DEFAULT_PARAM_VALUES: ParamValues = {
  temperature: 0.7,
  top_p: 0.8,
  top_k: 20,
  repetition_penalty: 1.05,
  max_tokens: 2048,
  system_prompt: '',
}

// ---------- 持久化（localStorage） ----------

interface StoredSession extends Session {
  messages: ChatMessage[]
}

interface SessionsSnapshot {
  sessions: StoredSession[]
  nextSessionId: number
  nextMessageId: number
}

const sessions = new Map<number, StoredSession>()
let nextSessionId = 1
let nextMessageId = 1
let paramValues: ParamValues = { ...DEFAULT_PARAM_VALUES }

try {
  const raw = localStorage.getItem(LS_SESSIONS)
  if (raw) {
    const snapshot = JSON.parse(raw) as SessionsSnapshot
    for (const s of snapshot.sessions) sessions.set(s.id, s)
    nextSessionId = snapshot.nextSessionId
    nextMessageId = snapshot.nextMessageId
  }
} catch {
  // 存档损坏时回退空状态，不阻塞界面
}

try {
  const raw = localStorage.getItem(LS_PARAMS)
  if (raw) paramValues = { ...DEFAULT_PARAM_VALUES, ...(JSON.parse(raw) as Partial<ParamValues>) }
} catch {
  // 同上
}

function persistSessions() {
  const snapshot: SessionsSnapshot = {
    sessions: [...sessions.values()],
    nextSessionId,
    nextMessageId,
  }
  localStorage.setItem(LS_SESSIONS, JSON.stringify(snapshot))
}

function persistParams() {
  localStorage.setItem(LS_PARAMS, JSON.stringify(paramValues))
}

// ---------- 模型状态机（内存态） ----------

let modelState: ModelState = 'unloaded'
let modelError: string | null = null
let generating = false

const now = () => Date.now() / 1000

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

const status = (): ModelStatus => ({
  state: modelState,
  vram_used_mb: modelState === 'loaded' ? 10500 : null,
  error: modelError,
})

// 非法迁移对齐契约：409 + { code, message }；mock 里以 reject ApiError 表示
const conflict = (message: string): ApiError => ({ code: 'illegal_transition', message })

const sessionNotFound = (sessionId: number): ApiError => ({
  code: 'SESSION_NOT_FOUND',
  message: `会话 ${sessionId} 不存在`,
})

const invalidParams = (message: string): ApiError => ({ code: 'INVALID_PARAMS', message })

// ---------- 参数：校验与预设匹配（对齐 server/params/presets.py） ----------

function validateParams(values: ParamValues): string | null {
  if (values.temperature < 0) return 'temperature 必须 ≥ 0'
  if (!(values.top_p > 0 && values.top_p <= 1)) return 'top_p 必须在 (0, 1] 区间'
  if (!Number.isInteger(values.top_k) || values.top_k < 0) return 'top_k 必须是非负整数'
  if (values.repetition_penalty <= 0) return 'repetition_penalty 必须 > 0'
  if (!Number.isInteger(values.max_tokens) || values.max_tokens < 1)
    return 'max_tokens 必须是正整数'
  return null
}

function matchPreset(values: SamplingValues): string {
  for (const [name, preset] of Object.entries(PRESETS)) {
    if (
      values.temperature === preset.temperature &&
      values.top_p === preset.top_p &&
      values.top_k === preset.top_k &&
      values.repetition_penalty === preset.repetition_penalty
    ) {
      return name
    }
  }
  return 'custom'
}

// ---------- 会话 ----------

// 首条消息自动命名：session_id 为空时直接以内容建会话；
// 已有会话标题仍为「新会话」（POST 建的空会话）时改用首条消息前 20 字（对齐 server/chat/engine.py）
function ensureSession(sessionId: number | null, content: string): StoredSession {
  if (sessionId !== null) {
    const existing = sessions.get(sessionId)
    if (!existing) throw sessionNotFound(sessionId)
    if (existing.title === DEFAULT_TITLE) {
      existing.title = content.slice(0, TITLE_LEN).trim() || DEFAULT_TITLE
    }
    return existing
  }
  const session: StoredSession = {
    id: nextSessionId++,
    title: content.slice(0, TITLE_LEN).trim() || DEFAULT_TITLE,
    created_at: now(),
    updated_at: now(),
    messages: [],
  }
  sessions.set(session.id, session)
  return session
}

function chat(req: ChatRequest, cb: ChatStreamCallbacks): ChatStreamHandle {
  if (modelState !== 'loaded') {
    queueMicrotask(() =>
      cb.onError({ code: 'model_not_loaded', message: '模型未加载，请先加载模型' }),
    )
    return { abort: () => undefined }
  }
  if (generating) {
    queueMicrotask(() => cb.onError({ code: 'conflict', message: '已有生成任务进行中' }))
    return { abort: () => undefined }
  }

  let session: StoredSession
  try {
    session = ensureSession(req.session_id, req.content)
  } catch (e) {
    // 会话不存在走 error 回调，与真实 client 的 HTTP 409 表现一致
    queueMicrotask(() => cb.onError(e as ApiError))
    return { abort: () => undefined }
  }

  generating = true
  const userMessage: ChatMessage = {
    id: nextMessageId++,
    role: 'user',
    content: req.content,
    created_at: now(),
  }
  const assistantMessage: ChatMessage = {
    id: nextMessageId++,
    role: 'assistant',
    content: '',
    created_at: now(),
  }
  session.messages.push(userMessage, assistantMessage)
  session.updated_at = now()
  persistSessions()

  const fullText = PRESET_REPLY.replace('{input}', req.content)
  // 按小片段切分模拟 token 流
  const tokens = fullText.match(/[\s\S]{1,3}/g) ?? []
  let index = 0
  let aborted = false

  const timer = setInterval(() => {
    if (aborted) {
      clearInterval(timer)
      generating = false
      persistSessions() // 部分结果保留入库（对齐后端停止语义）
      return
    }
    if (index === 0) {
      const meta = {
        session_id: session.id,
        user_message_id: userMessage.id,
        assistant_message_id: assistantMessage.id,
        title: session.title,
      }
      cb.onFrame?.({ time: Date.now() / 1000, event: 'meta', data: JSON.stringify(meta) })
      cb.onMeta(meta)
    }
    if (index < tokens.length) {
      const text = tokens[index]!
      assistantMessage.content += text
      cb.onFrame?.({ time: Date.now() / 1000, event: 'token', data: JSON.stringify({ text }) })
      cb.onToken({ text })
      index++
    }
    if (index >= tokens.length) {
      clearInterval(timer)
      generating = false
      // input_tokens 为近似值（对齐后端：仅估算），取历史消息总字符数的一半
      const totalChars = session.messages.reduce((sum, m) => sum + m.content.length, 0)
      const done = {
        usage: { input_tokens: Math.ceil(totalChars / 2), output_tokens: tokens.length },
      }
      persistSessions()
      cb.onFrame?.({ time: Date.now() / 1000, event: 'done', data: JSON.stringify(done) })
      cb.onDone(done)
    }
  }, TOKEN_INTERVAL_MS)

  return {
    abort: () => {
      aborted = true
    },
  }
}

export const api: LlmApi = {
  chat,
  stopChat: async () => undefined,
  listSessions: async (): Promise<SessionSummary[]> =>
    [...sessions.values()]
      .map(({ id, title, updated_at }) => ({ id, title, updated_at }))
      .sort((a, b) => b.updated_at - a.updated_at),
  createSession: async (title) => {
    const session: StoredSession = {
      id: nextSessionId++,
      title: title?.trim() || DEFAULT_TITLE,
      created_at: now(),
      updated_at: now(),
      messages: [],
    }
    sessions.set(session.id, session)
    persistSessions()
    const { messages: _messages, ...summary } = session
    return summary
  },
  getSessionMessages: async (sessionId) => {
    const session = sessions.get(sessionId)
    if (!session) throw sessionNotFound(sessionId)
    return [...session.messages]
  },
  renameSession: async (sessionId, title) => {
    const trimmed = title.trim()
    if (!trimmed) throw { code: 'EMPTY_TITLE', message: '标题不能为空' } satisfies ApiError
    const session = sessions.get(sessionId)
    if (!session) throw sessionNotFound(sessionId)
    session.title = trimmed // 重命名不推进 updated_at（对齐后端）
    persistSessions()
    const { messages: _messages, ...rest } = session
    return rest
  },
  deleteSession: async (sessionId) => {
    if (!sessions.delete(sessionId)) throw sessionNotFound(sessionId)
    persistSessions()
  },
  getParams: async () => ({ ...paramValues, preset: matchPreset(paramValues) }),
  saveParams: async (patch) => {
    const { preset, ...fields } = patch
    const values: ParamValues = { ...paramValues }
    // preset 服务端展开，显式字段优先于预设值（对齐后端）
    if (preset !== undefined) {
      const presetValues = PRESETS[preset]
      if (!presetValues) {
        throw invalidParams(`未知预设 ${preset}，可选：${Object.keys(PRESETS)}`)
      }
      Object.assign(values, presetValues)
    }
    const unknown = Object.keys(fields).filter(
      (key) => !(key in DEFAULT_PARAM_VALUES),
    )
    if (unknown.length > 0) throw invalidParams(`未知参数字段：${unknown}`)
    Object.assign(values, fields)
    const error = validateParams(values)
    if (error) throw invalidParams(error)
    paramValues = values
    persistParams()
    const matched = matchPreset(values)
    return {
      ...values,
      preset: preset !== undefined && preset === matched ? preset : matched,
    }
  },
  getModelStatus: async () => status(),
  loadModel: async () => {
    if (modelState !== 'unloaded' && modelState !== 'error') throw conflict('当前状态不允许加载')
    modelState = 'loading'
    modelError = null
    await delay(LOAD_MS)
    modelState = 'loaded'
    return status()
  },
  unloadModel: async () => {
    if (generating) throw conflict('生成中，禁止卸载，请先停止生成')
    if (modelState !== 'loaded') throw conflict('当前状态不允许卸载')
    modelState = 'unloading'
    await delay(UNLOAD_MS)
    modelState = 'unloaded'
    return status()
  },
}
