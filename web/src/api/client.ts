// 真实 HTTP/SSE 客户端（VITE_API_MODE=real 时使用，经 vite 代理到后端 :8000）
// SSE 帧解析与错误体解析在 api/sse.ts
import type {
  ApiError,
  ChatRequest,
  ChatSseEvent,
  ChatStreamCallbacks,
  ChatStreamHandle,
  CompressionRecord,
  InferenceParams,
  KnowledgeDocument,
  LlmApi,
  ModelStatus,
  Observability,
  Session,
  SessionMessages,
  SessionSummary,
} from './types'
import { parseErrorBody, postSse } from './sse'

const BASE = '/api'

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

function chat(req: ChatRequest, cb: ChatStreamCallbacks): ChatStreamHandle {
  return postSse('/chat', req, {
    onFrame: (frame) => cb.onFrame?.(frame),
    onEvent: (event, data) => {
      switch (event as ChatSseEvent) {
        case 'meta':
          cb.onMeta(data as never)
          break
        case 'token':
          cb.onToken(data as never)
          break
        case 'tool_call':
          cb.onToolCall?.(data as never)
          break
        case 'done':
          cb.onDone(data as never)
          break
        case 'error':
          cb.onError(data as ApiError)
          break
      }
    },
    onError: cb.onError,
  })
}

export const api: LlmApi = {
  chat,
  stopChat: (sessionId) =>
    request<void>('/chat/stop', jsonInit('POST', { session_id: sessionId })),
  listSessions: () => request<SessionSummary[]>('/sessions'),
  createSession: (title) => request<Session>('/sessions', jsonInit('POST', { title })),
  getSessionMessages: (sessionId) =>
    request<SessionMessages>(`/sessions/${sessionId}/messages`),
  renameSession: (sessionId, title) =>
    request<Session>(`/sessions/${sessionId}`, jsonInit('PATCH', { title })),
  deleteSession: (sessionId) => request<void>(`/sessions/${sessionId}`, { method: 'DELETE' }),
  getObservability: (sessionId, messageId) =>
    request<Observability>(`/sessions/${sessionId}/messages/${messageId}/observability`),
  compressSession: (sessionId) =>
    request<CompressionRecord>(`/sessions/${sessionId}/compress`, { method: 'POST' }),
  setRagEnabled: (sessionId, enabled) =>
    request<Session>(`/sessions/${sessionId}`, jsonInit('PATCH', { rag_enabled: enabled })),
  setToolEnabled: (sessionId, enabled) =>
    request<Session>(`/sessions/${sessionId}`, jsonInit('PATCH', { tool_enabled: enabled })),
  listDocuments: () => request<KnowledgeDocument[]>('/knowledge/documents'),
  uploadDocument: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request<KnowledgeDocument>('/knowledge/documents', {
      method: 'POST',
      body: form,
    })
  },
  deleteDocument: (documentId) =>
    request<void>(`/knowledge/documents/${documentId}`, { method: 'DELETE' }),
  getParams: () => request<InferenceParams>('/params'),
  saveParams: (patch) => request<InferenceParams>('/params', jsonInit('PUT', patch)),
  getModelStatus: () => request<ModelStatus>('/model/status'),
  loadModel: (modelId) =>
    request<ModelStatus>('/model/load', jsonInit('POST', modelId ? { model_id: modelId } : {})),
  unloadModel: () => request<ModelStatus>('/model/unload', { method: 'POST' }),
}
