// 会话状态：列表 + 当前会话 + 当前消息（技术方案 §1.2；契约见 docs/后端m3实现.md §2.1）
import { acceptHMRUpdate, defineStore } from 'pinia'
import { ref } from 'vue'
import { api, type ChatMessage, type ChatMeta, type SessionSummary } from '../api'

const DEFAULT_TITLE = '新会话'

// 本地临时 id 用负数，与后端自增 id 区分；meta 帧到达后用真实 id 对齐
let tempId = -1

export const useSessionStore = defineStore('session', () => {
  const sessions = ref<SessionSummary[]>([])
  const sessionId = ref<number | null>(null)
  const title = ref(DEFAULT_TITLE)
  const messages = ref<ChatMessage[]>([])
  // M6.5：当前会话最新压缩记录覆盖到的最大消息 id（无摘要为 null）
  const summaryCoveredUpto = ref<number | null>(null)
  // M7：当前会话的知识库检索开关（新建会话默认关）
  const ragEnabled = ref(false)
  // M9：当前会话的网页搜索工具开关（新建会话默认关）
  const toolEnabled = ref(false)

  function select(id: number | null, sessionTitle: string, msgs: ChatMessage[], coveredUpto: number | null = null, rag: boolean = false, tool: boolean = false) {
    sessionId.value = id
    title.value = sessionTitle
    messages.value = msgs
    summaryCoveredUpto.value = coveredUpto
    ragEnabled.value = rag
    toolEnabled.value = tool
  }

  // 列表项提到最前（列表按 updated_at 倒序，最新消息的会话排第一）
  function bumpSummary(id: number, patch?: Partial<SessionSummary>) {
    const index = sessions.value.findIndex((s) => s.id === id)
    if (index < 0) return
    const [item] = sessions.value.splice(index, 1)
    sessions.value.unshift({ ...item!, ...patch })
  }

  // ---------- 列表 CRUD（错误抛给调用方提示；SESSION_NOT_FOUND 走 handleSessionGone） ----------

  // 启动时加载列表并恢复最近会话（updated_at 最新者排第一）；无会话则保持空白待聊状态
  async function loadSessions() {
    sessions.value = await api.listSessions()
    const latest = sessions.value[0]
    if (latest) await openSession(latest.id)
  }

  // 新建空会话并选中；首次 chat 时后端用首条消息前 20 字自动命名
  async function createSession() {
    const session = await api.createSession()
    sessions.value.unshift({
      id: session.id,
      title: session.title,
      updated_at: session.updated_at,
      rag_enabled: session.rag_enabled,
      tool_enabled: session.tool_enabled,
    })
    select(session.id, session.title, [], null, session.rag_enabled, session.tool_enabled)
  }

  async function openSession(id: number) {
    const resp = await api.getSessionMessages(id)
    const summary = sessions.value.find((s) => s.id === id)
    select(id, summary?.title ?? DEFAULT_TITLE, resp.messages, resp.summary_covered_upto, summary?.rag_enabled ?? false, summary?.tool_enabled ?? false)
  }

  // M7：切换当前会话的知识库检索开关，列表项同步
  async function setRagEnabled(enabled: boolean) {
    if (sessionId.value === null) return
    await api.setRagEnabled(sessionId.value, enabled)
    ragEnabled.value = enabled
    const item = sessions.value.find((s) => s.id === sessionId.value)
    if (item) item.rag_enabled = enabled
  }

  // M9：切换当前会话的网页搜索工具开关，列表项同步（失败回弹同款）
  async function setToolEnabled(enabled: boolean) {
    if (sessionId.value === null) return
    await api.setToolEnabled(sessionId.value, enabled)
    toolEnabled.value = enabled
    const item = sessions.value.find((s) => s.id === sessionId.value)
    if (item) item.tool_enabled = enabled
  }

  // 一轮对话结束后刷新摘要覆盖范围（本轮可能触发了压缩）；只更新游标，不动消息列表
  async function refreshCoverage() {
    if (sessionId.value === null) return
    const resp = await api.getSessionMessages(sessionId.value)
    summaryCoveredUpto.value = resp.summary_covered_upto
  }

  async function rename(id: number, newTitle: string) {
    const session = await api.renameSession(id, newTitle)
    // 后端重命名不推进 updated_at，原位更新标题即可，无需重排
    const item = sessions.value.find((s) => s.id === id)
    if (item) item.title = session.title
    if (sessionId.value === id) title.value = session.title
  }

  // 删除后若删的是当前会话：切到剩余最近会话，无剩余则清空视图回到待聊状态
  async function remove(id: number) {
    await api.deleteSession(id)
    sessions.value = sessions.value.filter((s) => s.id !== id)
    if (sessionId.value !== id) return
    const next = sessions.value[0]
    if (next) await openSession(next.id)
    else reset()
  }

  // SESSION_NOT_FOUND 兜底：列表剔除该会话，若正打开则清空视图（引导用户新建）
  function handleSessionGone(id: number) {
    sessions.value = sessions.value.filter((s) => s.id !== id)
    if (sessionId.value === id) reset()
  }

  // chat 的 meta 帧回填：对齐 session_id 与标题（自动命名后列表同步更新）
  function applyChatMeta(meta: ChatMeta) {
    sessionId.value = meta.session_id
    title.value = meta.title
    if (sessions.value.some((s) => s.id === meta.session_id)) {
      bumpSummary(meta.session_id, { title: meta.title, updated_at: Date.now() / 1000 })
    } else {
      // 此前未建会话（首条消息直接开聊）：新会话检索/工具开关默认关
      sessions.value.unshift({
        id: meta.session_id,
        title: meta.title,
        updated_at: Date.now() / 1000,
        rag_enabled: false,
        tool_enabled: false,
      })
    }
  }

  // ---------- 当前消息 ----------

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
    select(null, DEFAULT_TITLE, [])
  }

  return {
    sessions,
    sessionId,
    title,
    messages,
    summaryCoveredUpto,
    ragEnabled,
    toolEnabled,
    loadSessions,
    createSession,
    openSession,
    refreshCoverage,
    setRagEnabled,
    setToolEnabled,
    rename,
    remove,
    handleSessionGone,
    applyChatMeta,
    appendUserMessage,
    appendAssistantPlaceholder,
    appendToken,
    markError,
    alignMessageId,
    reset,
  }
})

// 让 Pinia store 支持热更新，避免 dev 下新旧代码混跑（生产无影响）
if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useSessionStore, import.meta.hot))
}
