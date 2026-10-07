// 聊天流唯一入口：SSE 生命周期、逐 token 追加、停止（技术方案 §1.2）
// 同时把每帧原始记录写入 debug store（调试抽屉「原始输出」）；meta 帧才携带消息 id，之前的帧先缓冲
import { ref } from 'vue'
import { api, type ChatMessage, type ChatStreamHandle, type SseFrame } from '../api'
import { useSessionStore } from '../stores/session'
import { useDebugStore } from '../stores/debug'

export function useChatStream() {
  const sessionStore = useSessionStore()
  const debugStore = useDebugStore()
  const streaming = ref(false)

  let handle: ChatStreamHandle | null = null
  let assistantMessage: ChatMessage | null = null
  let frameBuffer: SseFrame[] = []
  let frameTargetId: number | null = null

  function cleanup() {
    streaming.value = false
    handle = null
    assistantMessage = null
    frameBuffer = []
    frameTargetId = null
  }

  function recordFrame(frame: SseFrame) {
    if (frameTargetId !== null) debugStore.append(frameTargetId, frame)
    else frameBuffer.push(frame)
  }

  function send(content: string) {
    if (streaming.value || !content.trim()) return
    streaming.value = true
    sessionStore.appendUserMessage(content)

    handle = api.chat(
      { session_id: sessionStore.sessionId, content },
      {
        onFrame: recordFrame,
        onMeta: (meta) => {
          sessionStore.sessionId = meta.session_id
          sessionStore.title = meta.title
          assistantMessage = sessionStore.appendAssistantPlaceholder()
          sessionStore.alignMessageId(assistantMessage, meta.assistant_message_id)
          frameTargetId = meta.assistant_message_id
          for (const frame of frameBuffer) debugStore.append(frameTargetId, frame)
          frameBuffer = []
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
