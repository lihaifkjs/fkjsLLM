// mock 实现：VITE_API_MODE=mock（默认）时使用，行为对齐后端契约（docs/后端m3实现.md §2）
// 会话/消息/参数用 localStorage 持久化，模拟后端重启不丢；模型状态机仍为内存态
import type {
  ApiError,
  ChatMessage,
  ChatRequest,
  ChatStreamCallbacks,
  ChatStreamHandle,
  CreateExperimentRequest,
  Experiment,
  ExperimentRun,
  ExperimentSummary,
  InferenceParams,
  LlmApi,
  ModelState,
  ModelStatus,
  Observability,
  PromptMessage,
  RunParams,
  RunStreamCallbacks,
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
const LS_LAB = 'fkjsllm.mock.lab'

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
}

// ---------- 持久化（localStorage） ----------

interface StoredSession extends Session {
  messages: ChatMessage[]
}

interface SessionsSnapshot {
  sessions: StoredSession[]
  nextSessionId: number
  nextMessageId: number
}

const sessions = new Map<number, StoredSession>()
let nextSessionId = 1
let nextMessageId = 1
let paramValues: ParamValues = { ...DEFAULT_PARAM_VALUES }
// 参数是否已有存档（M5 params_snapshot.source 标注：stored=用户已保存 / default=全局默认）
let paramsStored = false

try {
  const raw = localStorage.getItem(LS_SESSIONS)
  if (raw) {
    const snapshot = JSON.parse(raw) as SessionsSnapshot
    for (const s of snapshot.sessions) sessions.set(s.id, s)
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
// 生成中的 assistant 消息 id：观测数据尚未落库，getObservability 对它返回全 null（对齐后端时序）
let generatingMessageId: number | null = null

const now = () => Date.now() / 1000

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

const status = (): ModelStatus => ({
  state: modelState,
  vram_used_mb: modelState === 'loaded' ? 10500 : null,
  error: modelError,
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
function buildObservability(session: StoredSession, assistantIndex: number): Observability {
  const assistant = session.messages[assistantIndex]!
  const current = session.messages[assistantIndex - 1]!
  const history = session.messages.slice(0, assistantIndex - 1)
  const systemContent = paramValues.system_prompt || 'You are a helpful assistant.'

  const system = estimateTokens(systemContent) + MSG_OVERHEAD_TOKENS
  const historyTokens = history.reduce(
    (sum, m) => sum + estimateTokens(m.content) + MSG_OVERHEAD_TOKENS,
    0,
  )
  const currentTokens = estimateTokens(current.content) + MSG_OVERHEAD_TOKENS
  const inputTokens = system + historyTokens + currentTokens
  const outputTokens = outputTokenCount(assistant.content)

  const promptMessages: PromptMessage[] = [
    { role: 'system', content: systemContent },
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
  }
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
    messages: [],
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
  persistSessions()

  const fullText = PRESET_REPLY.replace('{input}', req.content)
  // 按小片段切分模拟 token 流
  const tokens = fullText.match(/[\s\S]{1,3}/g) ?? []
  let index = 0
  let aborted = false

  const timer = setInterval(() => {
    if (aborted) {
      clearInterval(timer)
      generating = false
      generatingMessageId = null // 中断路径后端同样回填观测数据（finally 汇合点），此后可查
      persistSessions() // 部分结果保留入库（对齐后端停止语义）
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

// ---------- 参数实验场（M6，localStorage 持久化对齐后端契约） ----------

// 存储形状即详情形状；列表项在读取时截取（prompt 前 50 字 + run_count）
interface StoredExperiment extends Experiment {
  system_prompt: string // 创建时的全局 system prompt 快照（决策 D：实验间可比性）
}

interface LabSnapshot {
  experiments: StoredExperiment[]
  nextExperimentId: number
  nextRunId: number
}

const experiments = new Map<number, StoredExperiment>()
let nextExperimentId = 1
let nextRunId = 1

try {
  const raw = localStorage.getItem(LS_LAB)
  if (raw) {
    const snapshot = JSON.parse(raw) as LabSnapshot
    for (const e of snapshot.experiments) {
      // 刷新时遗留的 running：mock 生成在内存中，重载即中断，纠正为 stopped（对齐后端读详情纠错语义）
      for (const run of e.runs) {
        if (run.status === 'running') run.status = 'stopped'
      }
      experiments.set(e.id, e)
    }
    nextExperimentId = snapshot.nextExperimentId
    nextRunId = snapshot.nextRunId
  }
} catch {
  // 存档损坏时回退空状态，不阻塞界面
}

function persistLab() {
  const snapshot: LabSnapshot = {
    experiments: [...experiments.values()],
    nextExperimentId,
    nextRunId,
  }
  localStorage.setItem(LS_LAB, JSON.stringify(snapshot))
}

const experimentNotFound = (id: number): ApiError => ({
  code: 'EXPERIMENT_NOT_FOUND',
  message: `实验 ${id} 不存在`,
})

const runNotFound = (runId: number): ApiError => ({
  code: 'RUN_NOT_FOUND',
  message: `实验运行 ${runId} 不存在`,
})

// 差异化输出：按 temperature 选风格模板（教学对比），长度差异由 max_tokens 截断产生（见下）
const LAB_STYLES = [
  {
    maxTemperature: 0.4,
    text: '【低温度 · 确定性回答】针对你的问题，给出直接、收敛的回答：要点如下——其一，先明确概念边界；其二，按步骤推导；其三，结论与前提一一对应。低温度下输出稳定，多次生成结果几乎一致，适合事实性问答。',
  },
  {
    maxTemperature: 0.9,
    text: '【中温度 · 平衡回答】这是一个兼顾准确与可读性的回答。先给出核心结论，再补充两处背景说明，最后给一个简短的例子帮助理解。中等温度下表述自然流畅，同时保持话题不发散，是日常对话的常用档位。',
  },
  {
    maxTemperature: Infinity,
    text: '【高温度 · 发散回答】让我们换个角度畅想一下——这个问题可以联想到许多有趣的场景：比喻、反例、甚至跨领域的类比都值得一试。高温度下用词更大胆、句式更多变，适合头脑风暴，但偶发跑题与事实性偏差需要人工复核。',
  },
]

function buildLabReply(params: RunParams, prompt: string): string {
  const style = LAB_STYLES.find((s) => params.temperature < s.maxTemperature)!
  return `${style.text}\n\n（本组参数：temperature=${params.temperature} / top_p=${params.top_p} / top_k=${params.top_k} / max_tokens=${params.max_tokens}；你的输入：「${prompt}」）`
}

// 实验单轮（system 快照 + 当前 prompt），history 恒为 0、prompt_eval_tokens = input_tokens（无 KV 复用）
// seg 三段之和 = input_tokens、decode_ms = output_tokens × TOKEN_INTERVAL_MS，与 done 帧自洽（M5 mock 同款约束）
function fillRunObservability(run: ExperimentRun, experiment: StoredExperiment) {
  const systemContent = experiment.system_prompt || 'You are a helpful assistant.'
  const system = estimateTokens(systemContent) + MSG_OVERHEAD_TOKENS
  const current = estimateTokens(experiment.prompt) + MSG_OVERHEAD_TOKENS
  const inputTokens = system + current
  const outputTokens = outputTokenCount(run.content)
  const promptMessages: PromptMessage[] = [
    { role: 'system', content: systemContent },
    { role: 'user', content: experiment.prompt },
  ]
  run.input_tokens = inputTokens
  run.output_tokens = outputTokens
  run.prompt_eval_tokens = inputTokens
  run.prompt_seg_tokens = { system, history: 0, current }
  run.prompt_messages = promptMessages
  run.prompt_text = renderPromptText(promptMessages)
  run.ttft_ms = TOKEN_INTERVAL_MS * 2
  run.prefill_ms = inputTokens * 1.5
  run.decode_ms = outputTokens * TOKEN_INTERVAL_MS
}

function toSummary(e: StoredExperiment): ExperimentSummary {
  return {
    id: e.id,
    prompt: e.prompt.slice(0, 50),
    created_at: e.created_at,
    run_count: e.runs.length,
  }
}

function toDetail(e: StoredExperiment): Experiment {
  return {
    id: e.id,
    prompt: e.prompt,
    system_prompt: e.system_prompt,
    created_at: e.created_at,
    runs: e.runs.map((run) => ({ ...run, params: { ...run.params } })),
  }
}

function generateRun(
  experimentId: number,
  runId: number,
  cb: RunStreamCallbacks,
): ChatStreamHandle {
  const fail = (error: ApiError): ChatStreamHandle => {
    queueMicrotask(() => cb.onError(error))
    return { abort: () => undefined }
  }
  const experiment = experiments.get(experimentId)
  if (!experiment) return fail(experimentNotFound(experimentId))
  const run = experiment.runs.find((r) => r.id === runId)
  if (!run) return fail(runNotFound(runId))
  if (modelState !== 'loaded') {
    return fail({ code: 'model_not_loaded', message: '模型未加载，请先加载模型' })
  }
  // 全局生成锁：与会话生成互斥（决策 E：409 GENERATION_BUSY，前端该栏标失败可重试）
  if (generating || run.status === 'running') {
    return fail({ code: 'GENERATION_BUSY', message: '已有生成任务进行中' })
  }

  generating = true
  run.status = 'running'
  run.content = ''
  run.error = null
  persistLab()

  // max_tokens 截断生效：不同参数组输出长度有差异（教学对比不失真）
  const fullText = buildLabReply(run.params, experiment.prompt)
  const tokens = (fullText.match(/[\s\S]{1,3}/g) ?? []).slice(0, run.params.max_tokens)
  let index = 0
  let aborted = false

  const timer = setInterval(() => {
    if (aborted) {
      clearInterval(timer)
      generating = false
      run.status = 'stopped' // 中断路径同样回填观测数据（对齐后端 finally 汇合点）
      fillRunObservability(run, experiment)
      persistLab()
      return
    }
    if (index === 0) cb.onMeta?.({ experiment_id: experimentId, run_id: runId })
    if (index < tokens.length) {
      const text = tokens[index]!
      run.content += text
      cb.onToken({ text })
      index++
    }
    if (index >= tokens.length) {
      clearInterval(timer)
      generating = false
      run.status = 'done'
      fillRunObservability(run, experiment)
      persistLab()
      // done 帧 usage/timing 与详情接口观测字段同口径
      cb.onDone({
        usage: { input_tokens: run.input_tokens ?? 0, output_tokens: run.output_tokens ?? 0 },
        timing: {
          ttft_ms: run.ttft_ms ?? 0,
          prefill_ms: run.prefill_ms ?? 0,
          decode_ms: run.decode_ms ?? 0,
        },
      })
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
      .map(({ id, title, updated_at }) => ({ id, title, updated_at }))
      .sort((a, b) => b.updated_at - a.updated_at),
  createSession: async (title) => {
    const session: StoredSession = {
      id: nextSessionId++,
      title: title?.trim() || DEFAULT_TITLE,
      created_at: now(),
      updated_at: now(),
      messages: [],
    }
    sessions.set(session.id, session)
    persistSessions()
    const { messages: _messages, ...summary } = session
    return summary
  },
  getSessionMessages: async (sessionId) => {
    const session = sessions.get(sessionId)
    if (!session) throw sessionNotFound(sessionId)
    return [...session.messages]
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
  loadModel: async () => {
    if (modelState !== 'unloaded' && modelState !== 'error') throw conflict('当前状态不允许加载')
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
  // ---------- M6 实验场 ----------
  listExperiments: async (): Promise<ExperimentSummary[]> =>
    [...experiments.values()].map(toSummary).sort((a, b) => b.created_at - a.created_at),
  createExperiment: async (req: CreateExperimentRequest) => {
    const prompt = req.prompt.trim()
    if (!prompt) throw invalidParams('prompt 不能为空')
    if (!Array.isArray(req.param_sets) || req.param_sets.length < 2 || req.param_sets.length > 3) {
      throw invalidParams('参数组数量须为 2–3')
    }
    for (const params of req.param_sets) {
      const error = validateParamFields(params)
      if (error) throw invalidParams(error)
    }
    const experiment: StoredExperiment = {
      id: nextExperimentId++,
      prompt,
      system_prompt: paramValues.system_prompt,
      created_at: now(),
      runs: req.param_sets.map((params, seq) => ({
        id: nextRunId++,
        seq,
        params: { ...params },
        status: 'pending',
        content: '',
        error: null,
        input_tokens: null,
        output_tokens: null,
        prompt_eval_tokens: null,
        prompt_text: null,
        prompt_messages: null,
        prompt_seg_tokens: null,
        ttft_ms: null,
        prefill_ms: null,
        decode_ms: null,
      })),
    }
    experiments.set(experiment.id, experiment)
    persistLab()
    return toDetail(experiment)
  },
  getExperiment: async (id) => {
    const experiment = experiments.get(id)
    if (!experiment) throw experimentNotFound(id)
    return toDetail(experiment)
  },
  deleteExperiment: async (id) => {
    if (!experiments.delete(id)) throw experimentNotFound(id)
    persistLab()
  },
  generateRun,
  stopRun: async (experimentId, runId) => {
    const experiment = experiments.get(experimentId)
    if (!experiment) throw experimentNotFound(experimentId)
    const run = experiment.runs.find((r) => r.id === runId)
    if (!run) throw runNotFound(runId)
    // 兜底：正常停止路径由 generateRun 的 abort 分支回填；这里只纠正状态
    if (run.status === 'running') {
      run.status = 'stopped'
      persistLab()
    }
  },
}
