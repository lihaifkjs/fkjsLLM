// 真实 HTTP/SSE 客户端（VITE_API_MODE=real 时使用，经 vite 代理到后端 :8000）
// EventSource 不支持 POST，SSE 用 fetch + ReadableStream 手动解析（技术方案 §5.4）
import type {
  ApiError,
  ChatMessage,
  ChatRequest,
  ChatStreamCallbacks,
  ChatStreamHandle,
  ChatSseEvent,
  InferenceParams,
  LlmApi,
  ModelStatus,
  Session,
  SessionSummary,
} from './types'

const BASE = '/api'

async function parseErrorBody(res: Response): Promise<ApiError> {
  try {
    // FastAPI 的 HTTPException 会把 {code,message} 包进 detail 字段（400 类错误），
    // ServiceError 全局处理器则直接返回顶层 {code,message}（409 类错误），两种都要解
    const body = (await res.json()) as {
      code?: string
      message?: string
      detail?: { code?: string; message?: string } | string
    }
    const payload = typeof body.detail === 'object' && body.detail !== null ? body.detail : body
    return {
      code: payload.code ?? `http_${res.status}`,
      message:
        payload.message ?? (typeof body.detail === 'string' ? body.detail : res.statusText),
    }
  } catch {
    return { code: `http_${res.status}`, message: res.statusText }
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, init)
  if (!res.ok) throw await parseErrorBody(res)
  return (await res.json()) as T
}

function jsonInit(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }
}

// 解析 SSE 帧：按空行分帧，帧内取 event: / data: 行
function parseSseFrames(buffer: string): { frames: { event: string; data: string }[]; rest: string } {
  const frames: { event: string; data: string }[] = []
  const parts = buffer.split('\n\n')
  const rest = parts.pop() ?? ''
  for (const part of parts) {
    let event = 'message'
    const dataLines: string[] = []
    for (const line of part.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim()
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
    }
    if (dataLines.length > 0) frames.push({ event, data: dataLines.join('\n') })
  }
  return { frames, rest }
}

function chat(req: ChatRequest, cb: ChatStreamCallbacks): ChatStreamHandle {
  const controller = new AbortController()
  void (async () => {
    try {
      const res = await fetch(`${BASE}/chat`, { ...jsonInit('POST', req), signal: controller.signal })
      if (!res.ok || !res.body) {
        cb.onError(res.body ? await parseErrorBody(res) : { code: 'no_body', message: '响应无流式内容' })
        return
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const { frames, rest } = parseSseFrames(buffer)
        buffer = rest
        for (const frame of frames) {
          cb.onFrame?.({ time: Date.now() / 1000, event: frame.event, data: frame.data })
          const data = JSON.parse(frame.data)
          switch (frame.event as ChatSseEvent) {
            case 'meta':
              cb.onMeta(data)
              break
            case 'token':
              cb.onToken(data)
              break
            case 'done':
              cb.onDone(data)
              break
            case 'error':
              cb.onError(data)
              break
          }
        }
      }
    } catch (e) {
      if (!controller.signal.aborted) {
        cb.onError({ code: 'network', message: e instanceof Error ? e.message : String(e) })
      }
    }
  })()
  return { abort: () => controller.abort() }
}

export const api: LlmApi = {
  chat,
  stopChat: (sessionId) =>
    request<void>('/chat/stop', jsonInit('POST', { session_id: sessionId })),
  listSessions: () => request<SessionSummary[]>('/sessions'),
  createSession: (title) => request<Session>('/sessions', jsonInit('POST', { title })),
  getSessionMessages: (sessionId) => request<ChatMessage[]>(`/sessions/${sessionId}/messages`),
  renameSession: (sessionId, title) =>
    request<Session>(`/sessions/${sessionId}`, jsonInit('PATCH', { title })),
  deleteSession: (sessionId) => request<void>(`/sessions/${sessionId}`, { method: 'DELETE' }),
  getParams: () => request<InferenceParams>('/params'),
  saveParams: (patch) => request<InferenceParams>('/params', jsonInit('PUT', patch)),
  getModelStatus: () => request<ModelStatus>('/model/status'),
  loadModel: () => request<ModelStatus>('/model/load', { method: 'POST' }),
  unloadModel: () => request<ModelStatus>('/model/unload', { method: 'POST' }),
}
