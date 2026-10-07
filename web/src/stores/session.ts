// 当前会话消息（M2 最小实现：只维护当前会话，列表 CRUD 留 M3）
import { defineStore } from 'pinia'
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

  function reset() {
    sessionId.value = null
    title.value = '新会话'
    messages.value = []
  }

  return { sessionId, title, messages, appendUserMessage, appendAssistantPlaceholder, appendToken, markError, reset }
})
