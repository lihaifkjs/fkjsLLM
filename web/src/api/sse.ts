// SSE 公共封装：chat 流式帧协议（meta/token/done/error）的 fetch + ReadableStream 手动解析
// EventSource 不支持 POST，统一用 fetch + ReadableStream 手动解析（技术方案 §5.4）
import type { ApiError, SseFrame } from './types'
import { authHeader, redirectToLogin } from '../utils/auth'

const BASE = '/api'

export async function parseErrorBody(res: Response): Promise<ApiError> {
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

// 解析 SSE 帧：按空行分帧，帧内取 event: / data: 行
export function parseSseFrames(buffer: string): {
  frames: { event: string; data: string }[]
  rest: string
} {
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

export interface PostSseOptions {
  // 每帧原始记录（可选，调试抽屉用）；data 不重新序列化，保真
  onFrame?: (frame: SseFrame) => void
  // 已解析帧：event 为帧类型，data 为 JSON.parse 后的载荷
  onEvent: (event: string, data: unknown) => void
  // HTTP 非 2xx（含 409 GENERATION_BUSY）与网络错误统一走这里
  onError: (error: ApiError) => void
}

// POST 一个 SSE 请求；返回的 abort() 只终止本地读取（服务端中断走各自的 stop 接口）
export function postSse(path: string, body: unknown, opts: PostSseOptions): { abort: () => void } {
  const controller = new AbortController()
  void (async () => {
    try {
      const res = await fetch(`${BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeader() },
        body: JSON.stringify(body),
        signal: controller.signal,
      })
      if (!res.ok || !res.body) {
        if (res.status === 401) redirectToLogin() // M10：token 失效跳登录页
        opts.onError(
          res.body ? await parseErrorBody(res) : { code: 'no_body', message: '响应无流式内容' },
        )
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
          opts.onFrame?.({ time: Date.now() / 1000, event: frame.event, data: frame.data })
          opts.onEvent(frame.event, JSON.parse(frame.data))
        }
      }
    } catch (e) {
      if (!controller.signal.aborted) {
        opts.onError({ code: 'network', message: e instanceof Error ? e.message : String(e) })
      }
    }
  })()
  return { abort: () => controller.abort() }
}
