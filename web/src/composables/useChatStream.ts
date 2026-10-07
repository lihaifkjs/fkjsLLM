// 聊天流唯一入口：SSE 生命周期、逐 token 追加、停止（技术方案 §1.2）
import { ref } from 'vue'
import { api, type ChatMessage, type ChatStreamHandle } from '../api'
import { useSessionStore } from '../stores/session'

export function useChatStream() {
  const sessionStore = useSessionStore()
  const streaming = ref(false)

  let handle: ChatStreamHandle | null = null
  let assistantMessage: ChatMessage | null = null

  function cleanup() {
    streaming.value = false
    handle = null
    assistantMessage = null
  }

  function send(content: string) {
    if (streaming.value || !content.trim()) return
    streaming.value = true
    sessionStore.appendUserMessage(content)

    handle = api.chat(
      { session_id: sessionStore.sessionId, content },
      {
        onMeta: (meta) => {
          sessionStore.sessionId = meta.session_id
          sessionStore.title = meta.title
          assistantMessage = sessionStore.appendAssistantPlaceholder()
        },
        onToken: (token) => {
          if (assistantMessage) sessionStore.appendToken(assistantMessage, token.text)
        },
        onDone: () => cleanup(),
        onError: (error) => {
          // 错误可能早于 meta（如模型未加载），此时尚无占位消息
          if (!assistantMessage) assistantMessage = sessionStore.appendAssistantPlaceholder()
          sessionStore.markError(assistantMessage, error.message)
          cleanup()
        },
      },
    )
  }

  function stop() {
    handle?.abort()
    if (sessionStore.sessionId !== null) void api.stopChat(sessionStore.sessionId)
    cleanup()
  }

  return { streaming, send, stop }
}
