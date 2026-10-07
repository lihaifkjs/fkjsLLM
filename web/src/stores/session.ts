// 当前会话消息（M2 最小实现：只维护当前会话，列表 CRUD 留 M3）
import { acceptHMRUpdate, defineStore } from 'pinia'
import { ref } from 'vue'
import type { ChatMessage } from '../api'

// 本地临时 id 用负数，与后端自增 id 区分；meta 帧到达后用真实 id 对齐
let tempId = -1

export const useSessionStore = defineStore('session', () => {
  const sessionId = ref<number | null>(null)
  const title = ref('新会话')
  const messages = ref<ChatMessage[]>([])

  // 返回响应式代理（数组内的元素），保证后续 appendToken/markError 的修改触发界面更新
  function appendUserMessage(content: string): ChatMessage {
    const message: ChatMessage = { id: tempId--, role: 'user', content, created_at: Date.now() / 1000 }
    messages.value.push(message)
    return messages.value[messages.value.length - 1]!
  }

  function appendAssistantPlaceholder(): ChatMessage {
    const message: ChatMessage = { id: tempId--, role: 'assistant', content: '', created_at: Date.now() / 1000 }
    messages.value.push(message)
    return messages.value[messages.value.length - 1]!
  }

  function appendToken(target: ChatMessage, text: string) {
    target.content += text
  }

  function markError(target: ChatMessage, message: string) {
    target.content = target.content ? `${target.content}\n\n> ⚠ ${message}` : `> ⚠ ${message}`
  }

  // meta 帧到达后把占位消息的临时 id 替换为后端真实 id（调试帧记录以真实 id 为键）
  function alignMessageId(target: ChatMessage, realId: number) {
    target.id = realId
  }

  function reset() {
    sessionId.value = null
    title.value = '新会话'
    messages.value = []
  }

  return { sessionId, title, messages, appendUserMessage, appendAssistantPlaceholder, appendToken, markError, alignMessageId, reset }
})

// 让 Pinia store 支持热更新，避免 dev 下新旧代码混跑（生产无影响）
if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useSessionStore, import.meta.hot))
}
