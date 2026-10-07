// 调试数据：按 assistant 消息 id 记录 SSE 原始帧序列（调试抽屉「原始输出」数据源）
// 独立于 session store：ChatMessage 是契约类型，不掺调试数据
import { acceptHMRUpdate, defineStore } from 'pinia'
import { reactive } from 'vue'
import type { SseFrame } from '../api'

export const useDebugStore = defineStore('debug', () => {
  const framesByMessage = reactive(new Map<number, SseFrame[]>())

  function append(messageId: number, frame: SseFrame) {
    let frames = framesByMessage.get(messageId)
    if (!frames) {
      frames = []
      framesByMessage.set(messageId, frames)
    }
    frames.push(frame)
  }

  function framesOf(messageId: number): SseFrame[] {
    return framesByMessage.get(messageId) ?? []
  }

  return { append, framesOf }
})

// 让 Pinia store 支持热更新，避免 dev 下新旧代码混跑（生产无影响）
if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useDebugStore, import.meta.hot))
}
