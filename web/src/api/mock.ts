// mock 实现：后端不可用期间使用，默认值见 docs/前端web.md §2
// 会话为内存存储，刷新即丢（M3 接真实持久化）
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

let modelState: ModelState = 'unloaded'
let modelError: string | null = null
let generating = false

let params: InferenceParams = {
  temperature: 0.7,
  top_p: 0.8,
  top_k: 20,
  repetition_penalty: 1.05,
  max_tokens: 2048,
  system_prompt: '',
  preset: '',
}

interface StoredSession extends Session {
  messages: ChatMessage[]
}

const sessions = new Map<number, StoredSession>()
let nextSessionId = 1
let nextMessageId = 1

const now = () => Date.now() / 1000

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

const status = (): ModelStatus => ({
  state: modelState,
  vram_used_mb: modelState === 'loaded' ? 10500 : null,
  error: modelError,
})

// 非法迁移对齐契约：409 + { code, message }；mock 里以 reject ApiError 表示
const conflict = (message: string): ApiError => ({ code: 'illegal_transition', message })

function ensureSession(sessionId: number | null, content: string): StoredSession {
  if (sessionId !== null) {
    const existing = sessions.get(sessionId)
    if (!existing) throw { code: 'not_found', message: `会话 ${sessionId} 不存在` } satisfies ApiError
    return existing
  }
  const session: StoredSession = {
    id: nextSessionId++,
    title: content.slice(0, 20) || '新会话',
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

  generating = true
  const session = ensureSession(req.session_id, req.content)
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

  const fullText = PRESET_REPLY.replace('{input}', req.content)
  // 按小片段切分模拟 token 流
  const tokens = fullText.match(/[\s\S]{1,3}/g) ?? []
  let index = 0
  let aborted = false

  const timer = setInterval(() => {
    if (aborted) {
      clearInterval(timer)
      generating = false
      return
    }
    if (index === 0) {
      cb.onMeta({
        session_id: session.id,
        user_message_id: userMessage.id,
        assistant_message_id: assistantMessage.id,
        title: session.title,
      })
    }
    if (index < tokens.length) {
      const text = tokens[index]!
      assistantMessage.content += text
      cb.onToken({ text })
      index++
    }
    if (index >= tokens.length) {
      clearInterval(timer)
      generating = false
      cb.onDone({
        usage: { input_tokens: Math.ceil(req.content.length / 2), output_tokens: tokens.length },
      })
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
      title: title || '新会话',
      created_at: now(),
      updated_at: now(),
      messages: [],
    }
    sessions.set(session.id, session)
    return { id: session.id, title: session.title, created_at: session.created_at, updated_at: session.updated_at }
  },
  getSessionMessages: async (sessionId) => sessions.get(sessionId)?.messages ?? [],
  getParams: async () => ({ ...params }),
  saveParams: async (next) => {
    params = { ...next }
    return { ...params }
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
