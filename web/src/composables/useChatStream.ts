// 聊天流唯一入口：SSE 生命周期、逐 token 追加、停止（技术方案 §1.2）
// 同时把每帧原始记录写入 debug store（调试抽屉「原始输出」）；meta 帧才携带消息 id，之前的帧先缓冲
// 轻量可观测性（PRD 3.4）：tokensPerSec 用已收到 token 帧数 / 经过时间实时计算；lastUsage 记 done 帧 usage
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api, type ChatMessage, type ChatStreamHandle, type SseFrame, type Usage } from '../api'
import { useSessionStore } from '../stores/session'
import { useDebugStore } from '../stores/debug'

export function useChatStream() {
  const sessionStore = useSessionStore()
  const debugStore = useDebugStore()
  const streaming = ref(false)
  const tokensPerSec = ref(0)
  const lastUsage = ref<Usage | null>(null)
  // M9：工具调用中间态（tool_call 事件置位，回答轮首个 token 到达清除）
  const searchQuery = ref<string | null>(null)

  let handle: ChatStreamHandle | null = null
  let assistantMessage: ChatMessage | null = null
  let frameBuffer: SseFrame[] = []
  let frameTargetId: number | null = null
  let tokenFrames = 0
  let firstTokenAt: number | null = null

  function cleanup() {
    streaming.value = false
    searchQuery.value = null
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
    tokensPerSec.value = 0
    tokenFrames = 0
    firstTokenAt = null
    sessionStore.appendUserMessage(content)

    handle = api.chat(
      { session_id: sessionStore.sessionId, content },
      {
        onFrame: recordFrame,
        onMeta: (meta) => {
          sessionStore.applyChatMeta(meta)
          assistantMessage = sessionStore.appendAssistantPlaceholder()
          sessionStore.alignMessageId(assistantMessage, meta.assistant_message_id)
          frameTargetId = meta.assistant_message_id
          for (const frame of frameBuffer) debugStore.append(frameTargetId, frame)
          frameBuffer = []
        },
        onToken: (token) => {
          searchQuery.value = null  // 回答轮 token 到达即搜索中间态结束
          if (assistantMessage) sessionStore.appendToken(assistantMessage, token.text)
          tokenFrames++
          const nowSec = Date.now() / 1000
          if (firstTokenAt === null) firstTokenAt = nowSec
          const elapsed = nowSec - firstTokenAt
          if (elapsed > 0) tokensPerSec.value = Math.round((tokenFrames / elapsed) * 10) / 10
        },
        onToolCall: (toolCall) => {
          searchQuery.value = toolCall.query
        },
        onDone: (done) => {
          lastUsage.value = done.usage
          cleanup()
          // 本轮可能触发了摘要压缩（M6.5）：刷新覆盖游标让有损标识立即出现
          void sessionStore.refreshCoverage()
        },
        onError: (error) => {
          // 会话在后端已不存在（如被删除）：提示并清空视图，引导新建
          if (error.code === 'SESSION_NOT_FOUND') {
            if (sessionStore.sessionId !== null) sessionStore.handleSessionGone(sessionStore.sessionId)
            ElMessage.warning('会话不存在或已被删除，请新建会话')
            cleanup()
            return
          }
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

  return { streaming, tokensPerSec, lastUsage, searchQuery, send, stop }
}
