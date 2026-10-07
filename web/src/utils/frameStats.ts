// 从 SSE 帧序列提取统计量：token 计数（done 帧 usage）与性能指标（帧时间戳差）
import type { SseFrame } from '../api'

export interface FrameStats {
  inputTokens: number | null // done 帧 usage.input_tokens（近似值，见后端实现.md §4.3）
  outputTokens: number | null
  tokenFrames: number // token 事件帧数
  ttftMs: number | null // 首 token 延迟：meta 帧 → 首个 token 帧
  totalMs: number | null // 全程耗时：meta 帧 → 最后一帧
  tokensPerSec: number | null // 输出 token 数 / token 阶段耗时
}

export function frameStats(frames: SseFrame[]): FrameStats {
  const meta = frames.find((f) => f.event === 'meta')
  const tokenFrames = frames.filter((f) => f.event === 'token')
  const done = frames.find((f) => f.event === 'done')
  const base = meta ?? frames[0]

  let inputTokens: number | null = null
  let outputTokens: number | null = null
  if (done) {
    try {
      const usage = (JSON.parse(done.data) as { usage?: { input_tokens?: number; output_tokens?: number } }).usage
      inputTokens = usage?.input_tokens ?? null
      outputTokens = usage?.output_tokens ?? null
    } catch {
      // data 非合法 JSON 时保持 null，面板降级显示
    }
  }

  const firstToken = tokenFrames[0]
  const lastFrame = frames[frames.length - 1]
  const lastToken = tokenFrames[tokenFrames.length - 1]

  const ttftMs = base && firstToken ? Math.round((firstToken.time - base.time) * 1000) : null
  const totalMs = base && lastFrame ? Math.round((lastFrame.time - base.time) * 1000) : null

  let tokensPerSec: number | null = null
  if (firstToken && lastToken && lastToken.time > firstToken.time) {
    const count = outputTokens ?? tokenFrames.length
    tokensPerSec = Math.round((count / (lastToken.time - firstToken.time)) * 10) / 10
  }

  return { inputTokens, outputTokens, tokenFrames: tokenFrames.length, ttftMs, totalMs, tokensPerSec }
}
