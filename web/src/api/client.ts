// 真实 HTTP/SSE 客户端（VITE_API_MODE=real 时使用，经 vite 代理到后端 :8000）
// SSE 帧解析与错误体解析在 api/sse.ts，chat 与实验场 generate 共用（M6 计划 §8）
import type {
  ApiError,
  ChatMessage,
  ChatRequest,
  ChatSseEvent,
  ChatStreamCallbacks,
  ChatStreamHandle,
  Experiment,
  ExperimentSummary,
  InferenceParams,
  LlmApi,
  ModelStatus,
  Observability,
  RunStreamCallbacks,
  Session,
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

// 实验 run 生成：帧协议与 chat 完全一致（M6 计划 §4.3），无 onFrame（实验不进调试抽屉）
function generateRun(
  experimentId: number,
  runId: number,
  cb: RunStreamCallbacks,
): ChatStreamHandle {
  return postSse(`/experiments/${experimentId}/runs/${runId}/generate`, {}, {
    onEvent: (event, data) => {
      switch (event as ChatSseEvent) {
        case 'meta':
          cb.onMeta?.(data as never)
          break
        case 'token':
          cb.onToken(data as never)
          break
        case 'done':
          cb.onDone(data as never)
          break
        case 'error':
          cb.onError(data as ApiError)
          break
      }
    },
    // HTTP 层错误（404 EXPERIMENT_NOT_FOUND / RUN_NOT_FOUND、409 GENERATION_BUSY）与帧 error 同通道
    onError: cb.onError,
  })
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
  getObservability: (sessionId, messageId) =>
    request<Observability>(`/sessions/${sessionId}/messages/${messageId}/observability`),
  getParams: () => request<InferenceParams>('/params'),
  saveParams: (patch) => request<InferenceParams>('/params', jsonInit('PUT', patch)),
  getModelStatus: () => request<ModelStatus>('/model/status'),
  loadModel: () => request<ModelStatus>('/model/load', { method: 'POST' }),
  unloadModel: () => request<ModelStatus>('/model/unload', { method: 'POST' }),
  listExperiments: () => request<ExperimentSummary[]>('/experiments'),
  createExperiment: (req) => request<Experiment>('/experiments', jsonInit('POST', req)),
  getExperiment: (id) => request<Experiment>(`/experiments/${id}`),
  deleteExperiment: (id) => request<void>(`/experiments/${id}`, { method: 'DELETE' }),
  generateRun,
  stopRun: (experimentId, runId) =>
    request<void>(`/experiments/${experimentId}/runs/${runId}/stop`, { method: 'POST' }),
}
