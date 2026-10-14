// mock 实现：VITE_API_MODE=mock（默认）时使用，行为对齐后端契约（docs/后端m3实现.md §2）
// 会话/消息/参数用 localStorage 持久化，模拟后端重启不丢；模型状态机仍为内存态
import type {
  ApiError,
  ChatMessage,
  ChatRequest,
  ChatStreamCallbacks,
  ChatStreamHandle,
  InferenceParams,
  KnowledgeDocument,
  LlmApi,
  ModelState,
  ModelStatus,
  Observability,
  PromptMessage,
  Session,
  SessionSummary,
} from './types'
import { validateParamFields } from '../utils/paramFields'

const TOKEN_INTERVAL_MS = 30
const LOAD_MS = 3000
const UNLOAD_MS = 1000
const DEFAULT_TITLE = '新会话'
const TITLE_LEN = 20 // 一期标题取首条消息前 20 字（对齐 server/chat/engine.py）

const LS_SESSIONS = 'fkjsllm.mock.sessions'
const LS_PARAMS = 'fkjsllm.mock.params'

// 预设回复：覆盖 Markdown、代码块，便于验证渲染
const PRESET_REPLY = `这是 mock 回复（后端就绪后切换 \`VITE_API_MODE=real\` 联调）。

支持 **Markdown** 渲染与代码高亮：

\`\`\`python
def hello(name: str) -> str:
    return f"Hello, {name}!"

print(hello("LAN LLM"))
\`\`\`

- 流式输出按 30ms/token 模拟
- 模型加载约 3s、卸载约 1s

> 你的输入是：「{input}」`

// 内置预设与默认值对齐 server/params/presets.py（预设只覆盖采样四参数）
type SamplingValues = Pick<
  InferenceParams,
  'temperature' | 'top_p' | 'top_k' | 'repetition_penalty'
>

const PRESETS: Record<string, SamplingValues> = {
  precise: { temperature: 0.1, top_p: 0.5, top_k: 10, repetition_penalty: 1.05 },
  balanced: { temperature: 0.7, top_p: 0.8, top_k: 20, repetition_penalty: 1.05 },
  creative: { temperature: 1.0, top_p: 0.9, top_k: 40, repetition_penalty: 1.05 },
}

type ParamValues = Omit<InferenceParams, 'preset'>

const DEFAULT_PARAM_VALUES: ParamValues = {
  temperature: 0.7,
  top_p: 0.8,
  top_k: 20,
  repetition_penalty: 1.05,
  max_tokens: 2048,
  system_prompt: '',
  context_strategy: 'sliding_window',
}

// M6.5：与 server/config.py 对齐的模拟常量
const MODEL_N_CTX = 8192
const SUMMARY_KEEP_ROUNDS = 4
const SUMMARY_PREFIX = '以下是此前对话的摘要（细节可能有损）：\n'
// 模拟压缩耗时：触发压缩的轮次首 token 前多停 25 帧（750ms）
const COMPRESS_DELAY_FRAMES = 25

// ---------- 持久化（localStorage） ----------

// M6.5：一次模拟压缩的记录（对齐后端 compressions 表口径）
interface StoredCompression {
  id: number
  triggerMessageId: number
  prevSummary: string
  sourceMessages: PromptMessage[]
  sourcePrompt: string
  summary: string
  coveredUpto: number
  sourceTokens: number
  summaryTokens: number
  inputTokens: number
  outputTokens: number
  timing: { ttft_ms: number; prefill_ms: number; decode_ms: number }
}

interface StoredSession extends Session {
  messages: ChatMessage[]
  // M6.5 新增字段；老存档无此字段，读取时归一化（见下方 load）
  compressions: StoredCompression[]
  coveredUpto: number | null
}

interface SessionsSnapshot {
  sessions: StoredSession[]
  nextSessionId: number
  nextMessageId: number
}

const sessions = new Map<number, StoredSession>()
let nextSessionId = 1
let nextMessageId = 1
let nextCompressionId = 1
let paramValues: ParamValues = { ...DEFAULT_PARAM_VALUES }
// 参数是否已有存档（M5 params_snapshot.source 标注：stored=用户已保存 / default=全局默认）
let paramsStored = false

try {
  const raw = localStorage.getItem(LS_SESSIONS)
  if (raw) {
    const snapshot = JSON.parse(raw) as SessionsSnapshot
    for (const s of snapshot.sessions) {
      // M6.5 前的老存档归一化
      s.compressions ??= []
      s.coveredUpto ??= null
      // M7 前的老存档归一化
      s.rag_enabled ??= false
      for (const c of s.compressions) {
        nextCompressionId = Math.max(nextCompressionId, c.id + 1)
      }
      sessions.set(s.id, s)
    }
    nextSessionId = snapshot.nextSessionId
    nextMessageId = snapshot.nextMessageId
  }
} catch {
  // 存档损坏时回退空状态，不阻塞界面
}

try {
  const raw = localStorage.getItem(LS_PARAMS)
  if (raw) {
    paramValues = { ...DEFAULT_PARAM_VALUES, ...(JSON.parse(raw) as Partial<ParamValues>) }
    paramsStored = true
  }
} catch {
  // 同上
}

function persistSessions() {
  const snapshot: SessionsSnapshot = {
    sessions: [...sessions.values()],
    nextSessionId,
    nextMessageId,
  }
  localStorage.setItem(LS_SESSIONS, JSON.stringify(snapshot))
}

function persistParams() {
  localStorage.setItem(LS_PARAMS, JSON.stringify(paramValues))
}

// ---------- 模型状态机（内存态） ----------

let modelState: ModelState = 'unloaded'
let modelError: string | null = null
let generating = false
// M8 最小桩：仅记录选中的模型条目，供 status 下发与选择器渲染
let modelId = 'qwen3-14b'
const AVAILABLE_MODELS = [
  { id: 'qwen3-14b', name: 'Qwen3-14B 原版' },
  { id: 'fkjs-14b', name: 'fkjs-14B（身份微调）' },
]
// 生成中的 assistant 消息 id：观测数据尚未落库，getObservability 对它返回全 null（对齐后端时序）
let generatingMessageId: number | null = null

const now = () => Date.now() / 1000

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

const status = (): ModelStatus => ({
  state: modelState,
  vram_used_mb: modelState === 'loaded' ? 10500 : null,
  error: modelError,
  n_ctx: MODEL_N_CTX,
  model_id: modelId,
  available_models: AVAILABLE_MODELS,
})

// 非法迁移对齐契约：409 + { code, message }；mock 里以 reject ApiError 表示
const conflict = (message: string): ApiError => ({ code: 'illegal_transition', message })

const sessionNotFound = (sessionId: number): ApiError => ({
  code: 'SESSION_NOT_FOUND',
  message: `会话 ${sessionId} 不存在`,
})

const invalidParams = (message: string): ApiError => ({ code: 'INVALID_PARAMS', message })

// M5：观测接口对不存在的消息返回 404 MESSAGE_NOT_FOUND（对齐后端首个 404 业务码）
const messageNotFound = (messageId: number): ApiError => ({
  code: 'MESSAGE_NOT_FOUND',
  message: `消息 ${messageId} 不存在`,
})

// ---------- 知识库文档（M7 最小桩：内存态，不复刻切块/检索行为） ----------

let nextDocumentId = 3
const documents: KnowledgeDocument[] = [
  { id: 1, filename: 'sqlite_notes.md', char_count: 342, chunk_count: 1, created_at: now() - 3600 },
  { id: 2, filename: 'llm_inference.md', char_count: 518, chunk_count: 2, created_at: now() - 600 },
]

// ---------- 参数：校验与预设匹配（对齐 server/params/presets.py） ----------

function validateParams(values: ParamValues): string | null {
  return validateParamFields(values)
}

function matchPreset(values: SamplingValues): string {
  for (const [name, preset] of Object.entries(PRESETS)) {
    if (
      values.temperature === preset.temperature &&
      values.top_p === preset.top_p &&
      values.top_k === preset.top_k &&
      values.repetition_penalty === preset.repetition_penalty
    ) {
      return name
    }
  }
  return 'custom'
}

// ---------- 观测数据（M5，对齐后端 observability 契约） ----------

// token 估算口径：≈字符数一半 + 每条消息 4 token 模板开销；输出按 30ms 流的分片数（与 done 帧一致）
const MSG_OVERHEAD_TOKENS = 4
const estimateTokens = (text: string) => Math.ceil(text.length / 2)
const outputTokenCount = (text: string) => text.match(/[\s\S]{1,3}/g)?.length ?? 0

// 渲染 ChatML 原文（含 <|im_start|> 等特殊 token，对齐 Qwen3 模板形状）
function renderPromptText(messages: PromptMessage[]): string {
  const body = messages.map((m) => `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`).join('')
  return `${body}<|im_start|>assistant\n`
}

// 构造某条 assistant 消息的观测假数据：seg 三段之和 = input_tokens，数值自洽；
// prompt_eval_tokens 在多轮时小于 input_tokens，模拟 KV 前缀复用（教学点）
// M6.5：摘要策略下，被覆盖历史不进上下文，替换为摘要段（对齐后端决策 E）
function buildObservability(session: StoredSession, assistantIndex: number): Observability {
  const assistant = session.messages[assistantIndex]!
  const current = session.messages[assistantIndex - 1]!
  const historyAll = session.messages.slice(0, assistantIndex - 1)
  const systemContent = paramValues.system_prompt || 'You are a helpful assistant.'

  // 该消息生成时生效的压缩（触发它的压缩对触发消息本身即生效）
  const active =
    paramValues.context_strategy === 'summary'
      ? session.compressions.filter((c) => c.triggerMessageId <= assistant.id).at(-1)
      : undefined
  const history = active
    ? historyAll.filter((m) => m.id > active.coveredUpto)
    : historyAll
  const summaryContent = active ? SUMMARY_PREFIX + active.summary : null

  const system =
    estimateTokens(systemContent) +
    MSG_OVERHEAD_TOKENS +
    (summaryContent ? estimateTokens(summaryContent) + MSG_OVERHEAD_TOKENS : 0)
  const historyTokens = history.reduce(
    (sum, m) => sum + estimateTokens(m.content) + MSG_OVERHEAD_TOKENS,
    0,
  )
  const currentTokens = estimateTokens(current.content) + MSG_OVERHEAD_TOKENS
  const inputTokens = system + historyTokens + currentTokens
  const outputTokens = outputTokenCount(assistant.content)

  const promptMessages: PromptMessage[] = [
    { role: 'system', content: systemContent },
    ...(summaryContent ? [{ role: 'system' as const, content: summaryContent }] : []),
    ...history.map((m) => ({ role: m.role, content: m.content })),
    { role: 'user' as const, content: current.content },
  ]

  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    // 有历史时模拟 KV 前缀复用：仅 system + 当前提问需要实际 prefill
    prompt_eval_tokens: historyTokens > 0 ? system + currentTokens : inputTokens,
    seg_tokens: { system, history: historyTokens, current: currentTokens },
    timing: {
      ttft_ms: TOKEN_INTERVAL_MS * 2,
      prefill_ms: inputTokens * 1.5,
      decode_ms: outputTokens * TOKEN_INTERVAL_MS,
    },
    params_snapshot: {
      ...paramValues,
      preset: matchPreset(paramValues),
      source: paramsStored ? 'stored' : 'default',
    },
    prompt_text: renderPromptText(promptMessages),
    prompt_messages: promptMessages,
    // M7 最小桩：会话开了检索开关就给一条固定假命中，不复刻真实检索行为
    retrieval: session.rag_enabled
      ? {
          query: current.content,
          hits: [
            {
              chunk_id: 1,
              document_id: 1,
              filename: 'sqlite_notes.md',
              score: 0.62,
              content:
                'WAL 模式（Write-Ahead Logging）允许多个读者与一个写者并发，读操作不再阻塞写操作。（mock 命中片段）',
            },
          ],
          injected_tokens: 64,
          latency_ms: 12.5,
        }
      : null,
    // 压缩三态：本消息触发（完整记录）/ 复用既有摘要 / 无
    compression: active
      ? active.triggerMessageId === assistant.id
        ? {
            triggered: true,
            id: active.id,
            prev_summary: active.prevSummary,
            source_messages: active.sourceMessages,
            source_prompt: active.sourcePrompt,
            summary: active.summary,
            covered_upto: active.coveredUpto,
            source_tokens: active.sourceTokens,
            summary_tokens: active.summaryTokens,
            input_tokens: active.inputTokens,
            output_tokens: active.outputTokens,
            timing: active.timing,
          }
        : { triggered: false, active_id: active.id }
      : null,
  }
}

// user 消息 / M5 前老消息的观测响应：各字段为 null（对齐后端契约）
function nullObservability(): Observability {
  return {
    input_tokens: null,
    output_tokens: null,
    prompt_eval_tokens: null,
    seg_tokens: null,
    timing: null,
    params_snapshot: null,
    prompt_text: null,
    prompt_messages: null,
    compression: null,
    retrieval: null,
  }
}

// ---------- 摘要压缩模拟（M6.5，对齐后端 chat/engine.py 语义） ----------

// 超限时把溢出历史压成假摘要并记录；无新增溢出（复用）或未超限返回 false
function maybeCompress(session: StoredSession, triggerMessageId: number): boolean {
  const budget = MODEL_N_CTX - paramValues.max_tokens
  const systemTokens = estimateTokens(paramValues.system_prompt || '')
  // prev 含本轮 user 消息，不含空 assistant 占位（对齐后端 exclude 语义）
  const prev = session.messages.slice(0, -1)
  const total =
    systemTokens + prev.reduce((sum, m) => sum + estimateTokens(m.content), 0)
  if (total <= budget) return false

  const keep = SUMMARY_KEEP_ROUNDS * 2
  if (prev.length <= keep + 1) return false
  const overflow = prev.slice(0, prev.length - keep)
  const covered = session.coveredUpto ?? 0
  const newOverflow = overflow.filter((m) => m.id > covered)
  if (newOverflow.length === 0) return false // 增量游标已覆盖：复用既有摘要

  const prevRecord = session.compressions.at(-1)
  const topics = newOverflow
    .filter((m) => m.role === 'user')
    .map((m) => m.content.slice(0, 12))
    .join('；')
  const summary =
    (prevRecord ? `${prevRecord.summary}；` : '') +
    `另有 ${newOverflow.length} 条历史：${topics}（mock 摘要）`
  const sourceTokens = newOverflow.reduce(
    (sum, m) => sum + estimateTokens(m.content) + MSG_OVERHEAD_TOKENS,
    0,
  )
  const summaryTokens = estimateTokens(summary) + MSG_OVERHEAD_TOKENS
  const sourceMessages: PromptMessage[] = newOverflow.map((m) => ({
    role: m.role,
    content: m.content,
  }))
  session.compressions.push({
    id: nextCompressionId++,
    triggerMessageId,
    prevSummary: prevRecord?.summary ?? '',
    sourceMessages,
    sourcePrompt: renderPromptText([
      { role: 'system', content: '你是对话压缩器。（mock 压缩指令）' },
      {
        role: 'user',
        content: `【既有摘要】\n${prevRecord?.summary ?? '（无）'}\n\n【新增对话】\n${sourceMessages
          .map((m) => `${m.role}: ${m.content}`)
          .join('\n')}`,
      },
    ]),
    summary,
    coveredUpto: overflow[overflow.length - 1]!.id,
    sourceTokens,
    summaryTokens,
    inputTokens: sourceTokens + estimateTokens(prevRecord?.summary ?? '') + 40,
    outputTokens: summaryTokens,
    timing: {
      ttft_ms: COMPRESS_DELAY_FRAMES * TOKEN_INTERVAL_MS,
      prefill_ms: sourceTokens * 1.5,
      decode_ms: summaryTokens * TOKEN_INTERVAL_MS,
    },
  })
  session.coveredUpto = overflow[overflow.length - 1]!.id
  return true
}

// ---------- 会话 ----------

// 首条消息自动命名：session_id 为空时直接以内容建会话；
// 已有会话标题仍为「新会话」（POST 建的空会话）时改用首条消息前 20 字（对齐 server/chat/engine.py）
function ensureSession(sessionId: number | null, content: string): StoredSession {
  if (sessionId !== null) {
    const existing = sessions.get(sessionId)
    if (!existing) throw sessionNotFound(sessionId)
    if (existing.title === DEFAULT_TITLE) {
      existing.title = content.slice(0, TITLE_LEN).trim() || DEFAULT_TITLE
    }
    return existing
  }
  const session: StoredSession = {
    id: nextSessionId++,
    title: content.slice(0, TITLE_LEN).trim() || DEFAULT_TITLE,
    created_at: now(),
    updated_at: now(),
    rag_enabled: false,
    messages: [],
    compressions: [],
    coveredUpto: null,
  }
  sessions.set(session.id, session)
  return session
}

function chat(req: ChatRequest, cb: ChatStreamCallbacks): ChatStreamHandle {
  if (modelState !== 'loaded') {
    queueMicrotask(() =>
      cb.onError({ code: 'model_not_loaded', message: '模型未加载，请先加载模型' }),
    )
    return { abort: () => undefined }
  }
  if (generating) {
    queueMicrotask(() => cb.onError({ code: 'conflict', message: '已有生成任务进行中' }))
    return { abort: () => undefined }
  }

  let session: StoredSession
  try {
    session = ensureSession(req.session_id, req.content)
  } catch (e) {
    // 会话不存在走 error 回调，与真实 client 的 HTTP 409 表现一致
    queueMicrotask(() => cb.onError(e as ApiError))
    return { abort: () => undefined }
  }

  generating = true
  const userMessage: ChatMessage = {
    id: nextMessageId++,
    role: 'user',
    content: req.content,
    created_at: now(),
  }
  const assistantMessage: ChatMessage = {
    id: nextMessageId++,
    role: 'assistant',
    content: '',
    created_at: now(),
  }
  session.messages.push(userMessage, assistantMessage)
  session.updated_at = now()
  generatingMessageId = assistantMessage.id
  // M6.5：摘要策略下超限则先「压缩」（模拟耗时体现在首 token 延迟）
  const compressedThisTurn =
    paramValues.context_strategy === 'summary' &&
    maybeCompress(session, assistantMessage.id)
  persistSessions()

  const fullText = PRESET_REPLY.replace('{input}', req.content)
  // 按小片段切分模拟 token 流
  const tokens = fullText.match(/[\s\S]{1,3}/g) ?? []
  let index = compressedThisTurn ? -COMPRESS_DELAY_FRAMES : 0
  let aborted = false

  const timer = setInterval(() => {
    if (aborted) {
      clearInterval(timer)
      generating = false
      generatingMessageId = null // 中断路径后端同样回填观测数据（finally 汇合点），此后可查
      persistSessions() // 部分结果保留入库（对齐后端停止语义）
      return
    }
    if (index < 0) {
      index++ // 压缩进行中：尚无 SSE 帧（对齐后端：meta 帧在压缩完成后才发出）
      return
    }
    if (index === 0) {
      const meta = {
        session_id: session.id,
        user_message_id: userMessage.id,
        assistant_message_id: assistantMessage.id,
        title: session.title,
      }
      cb.onFrame?.({ time: Date.now() / 1000, event: 'meta', data: JSON.stringify(meta) })
      cb.onMeta(meta)
    }
    if (index < tokens.length) {
      const text = tokens[index]!
      assistantMessage.content += text
      cb.onFrame?.({ time: Date.now() / 1000, event: 'token', data: JSON.stringify({ text }) })
      cb.onToken({ text })
      index++
    }
    if (index >= tokens.length) {
      clearInterval(timer)
      generating = false
      generatingMessageId = null // done 后观测数据视为已落库
      // M5：usage 与 timing 复用 buildObservability 口径（assistant 消息恒为最后一条），保证观测接口数据自洽
      const obs = buildObservability(session, session.messages.length - 1)
      const done = {
        usage: { input_tokens: obs.input_tokens ?? 0, output_tokens: obs.output_tokens ?? 0 },
        timing: obs.timing ?? { ttft_ms: 0, prefill_ms: 0, decode_ms: 0 },
      }
      persistSessions()
      cb.onFrame?.({ time: Date.now() / 1000, event: 'done', data: JSON.stringify(done) })
      cb.onDone(done)
    }
  }, TOKEN_INTERVAL_MS)

  return {
    abort: () => {
      aborted = true
    },
  }
}

export const api: LlmApi = {
  chat,
  stopChat: async () => undefined,
  listSessions: async (): Promise<SessionSummary[]> =>
    [...sessions.values()]
      .map(({ id, title, updated_at, rag_enabled }) => ({ id, title, updated_at, rag_enabled }))
      .sort((a, b) => b.updated_at - a.updated_at),
  createSession: async (title) => {
    const session: StoredSession = {
      id: nextSessionId++,
      title: title?.trim() || DEFAULT_TITLE,
      created_at: now(),
      updated_at: now(),
      rag_enabled: false,
      messages: [],
      compressions: [],
      coveredUpto: null,
    }
    sessions.set(session.id, session)
    persistSessions()
    const { messages: _messages, ...summary } = session
    return summary
  },
  getSessionMessages: async (sessionId) => {
    const session = sessions.get(sessionId)
    if (!session) throw sessionNotFound(sessionId)
    // 批次4：存在压缩记录即下发游标（与当前策略无关，滑窗下也显示有损标识）
    return {
      messages: [...session.messages],
      summary_covered_upto: session.coveredUpto,
    }
  },
  // 手动压缩（批次4）最小桩：写死一条假记录，covered_upto 推到最新消息 id
  compressSession: async (sessionId) => {
    const session = sessions.get(sessionId)
    if (!session) throw sessionNotFound(sessionId)
    const last = session.messages.at(-1)
    if (!last) throw { code: 'EMPTY_SESSION', message: '会话没有消息，无可压缩历史' } satisfies ApiError
    const prevRecord = session.compressions.at(-1)
    const record: StoredCompression = {
      id: nextCompressionId++,
      triggerMessageId: 0, // 手动压缩无对应触发消息
      prevSummary: prevRecord?.summary ?? '',
      sourceMessages: [],
      sourcePrompt: '',
      summary: `手动压缩全部历史共 ${session.messages.length} 条（mock 摘要）`,
      coveredUpto: last.id,
      sourceTokens: 0,
      summaryTokens: 0,
      inputTokens: 0,
      outputTokens: 0,
      timing: { ttft_ms: 0, prefill_ms: 0, decode_ms: 0 },
    }
    session.compressions.push(record)
    session.coveredUpto = last.id
    persistSessions()
    return {
      triggered: true as const,
      id: record.id,
      prev_summary: record.prevSummary,
      source_messages: record.sourceMessages,
      source_prompt: record.sourcePrompt,
      summary: record.summary,
      covered_upto: record.coveredUpto,
      source_tokens: record.sourceTokens,
      summary_tokens: record.summaryTokens,
      input_tokens: record.inputTokens,
      output_tokens: record.outputTokens,
      timing: record.timing,
    }
  },
  getObservability: async (sessionId, messageId) => {
    const session = sessions.get(sessionId)
    if (!session) throw sessionNotFound(sessionId)
    const index = session.messages.findIndex((m) => m.id === messageId)
    if (index < 0) throw messageNotFound(messageId)
    // user 消息 / 生成中（尚未落库）无观测数据（对齐后端：各字段 null）；
    // assistant 消息必跟随一条 user（index ≥ 1）
    if (session.messages[index]!.role !== 'assistant' || index < 1 || messageId === generatingMessageId) {
      return nullObservability()
    }
    return buildObservability(session, index)
  },
  renameSession: async (sessionId, title) => {
    const trimmed = title.trim()
    if (!trimmed) throw { code: 'EMPTY_TITLE', message: '标题不能为空' } satisfies ApiError
    const session = sessions.get(sessionId)
    if (!session) throw sessionNotFound(sessionId)
    session.title = trimmed // 重命名不推进 updated_at（对齐后端）
    persistSessions()
    const { messages: _messages, ...rest } = session
    return rest
  },
  deleteSession: async (sessionId) => {
    if (!sessions.delete(sessionId)) throw sessionNotFound(sessionId)
    persistSessions()
  },
  setRagEnabled: async (sessionId, enabled) => {
    const session = sessions.get(sessionId)
    if (!session) throw sessionNotFound(sessionId)
    session.rag_enabled = enabled
    persistSessions()
    const { messages: _messages, ...rest } = session
    return rest
  },
  listDocuments: async () => [...documents].sort((a, b) => b.id - a.id),
  uploadDocument: async (file) => {
    const doc: KnowledgeDocument = {
      id: nextDocumentId++,
      filename: file.name,
      char_count: file.size,
      chunk_count: 1,
      created_at: now(),
    }
    documents.push(doc)
    return doc
  },
  deleteDocument: async (documentId) => {
    const index = documents.findIndex((d) => d.id === documentId)
    if (index < 0) {
      throw { code: 'DOCUMENT_NOT_FOUND', message: `文档 ${documentId} 不存在` } satisfies ApiError
    }
    documents.splice(index, 1)
  },
  getParams: async () => ({ ...paramValues, preset: matchPreset(paramValues) }),
  saveParams: async (patch) => {
    const { preset, ...fields } = patch
    const values: ParamValues = { ...paramValues }
    // preset 服务端展开，显式字段优先于预设值（对齐后端）
    if (preset !== undefined) {
      const presetValues = PRESETS[preset]
      if (!presetValues) {
        throw invalidParams(`未知预设 ${preset}，可选：${Object.keys(PRESETS)}`)
      }
      Object.assign(values, presetValues)
    }
    const unknown = Object.keys(fields).filter(
      (key) => !(key in DEFAULT_PARAM_VALUES),
    )
    if (unknown.length > 0) throw invalidParams(`未知参数字段：${unknown}`)
    Object.assign(values, fields)
    const error = validateParams(values)
    if (error) throw invalidParams(error)
    paramValues = values
    paramsStored = true
    persistParams()
    const matched = matchPreset(values)
    return {
      ...values,
      preset: preset !== undefined && preset === matched ? preset : matched,
    }
  },
  getModelStatus: async () => status(),
  loadModel: async (newModelId?: string) => {
    if (modelState !== 'unloaded' && modelState !== 'error') throw conflict('当前状态不允许加载')
    if (newModelId) modelId = newModelId
    modelState = 'loading'
    modelError = null
    await delay(LOAD_MS)
    modelState = 'loaded'
    return status()
  },
  unloadModel: async () => {
    if (generating) throw conflict('生成中，禁止卸载，请先停止生成')
    if (modelState !== 'loaded') throw conflict('当前状态不允许卸载')
    modelState = 'unloading'
    await delay(UNLOAD_MS)
    modelState = 'unloaded'
    return status()
  },
}
