// 参数实验场（M6）：实验列表 + 当前实验 + 串行队列状态机（决策 C：前端逐个 run 发 generate，后端不建队列）
// run 状态映射 UI：pending 排队中 / running 生成中 / done 完成 / error 失败 / stopped 已停止
// 观测数据随 GET /experiments/{id} 详情下发（决策 B）：每个 run 结束后拉详情回填 seg/prompt 等全量字段
import { acceptHMRUpdate, defineStore } from 'pinia'
import { ref } from 'vue'
import {
  api,
  type ChatStreamHandle,
  type Experiment,
  type ExperimentRun,
  type ExperimentSummary,
  type RunParams,
} from '../api'

interface ActiveRun {
  handle: ChatStreamHandle
  // 本地 abort 不会触发 onDone/onError（连接是本地掐断的），用 settle 收尾 promise
  settle: () => void
}

export const useLabStore = defineStore('lab', () => {
  const experiments = ref<ExperimentSummary[]>([])
  const current = ref<Experiment | null>(null)
  const queueRunning = ref(false)

  const activeRuns = new Map<number, ActiveRun>()

  async function loadExperiments() {
    experiments.value = await api.listExperiments()
  }

  async function createExperiment(prompt: string, runs: RunParams[]) {
    const experiment = await api.createExperiment({ prompt, param_sets: runs })
    current.value = experiment
    experiments.value = await api.listExperiments()
  }

  async function openExperiment(id: number) {
    const detail = await api.getExperiment(id)
    // 兜底纠正：running 但本地无活动句柄（刷新/断连后），不能永久显示「生成中」（计划 §8）
    // 后端读详情时会把 running 纠正为 error/stopped，此处再兜一层防漂移
    for (const run of detail.runs) {
      if (run.status === 'running' && !activeRuns.has(run.id)) {
        run.status = 'error'
        run.error = { code: 'INTERRUPTED', message: '生成已中断（页面刷新或连接断开）' }
      }
    }
    current.value = detail
  }

  async function removeExperiment(id: number) {
    await api.deleteExperiment(id)
    experiments.value = experiments.value.filter((e) => e.id !== id)
    if (current.value?.id === id) current.value = null
  }

  // 单个 run 的完整生命周期：done/error/stopped 三路径都会收尾并拉详情回填观测
  async function runOne(run: ExperimentRun) {
    const experimentId = current.value!.id
    run.status = 'running'
    run.error = null
    run.content = ''
    let failed: { code: string; message: string } | null = null
    await new Promise<void>((resolve) => {
      const handle = api.generateRun(experimentId, run.id, {
        onToken: (token) => {
          run.content += token.text
        },
        onDone: (done) => {
          run.status = 'done'
          run.input_tokens = done.usage.input_tokens
          run.output_tokens = done.usage.output_tokens
          run.ttft_ms = done.timing.ttft_ms
          run.prefill_ms = done.timing.prefill_ms
          run.decode_ms = done.timing.decode_ms
          activeRuns.delete(run.id)
          resolve()
        },
        // 含 409 GENERATION_BUSY（决策 E：该栏标失败、可重试，不重排队列）
        onError: (error) => {
          run.status = 'error'
          run.error = error
          failed = error
          activeRuns.delete(run.id)
          resolve()
        },
      })
      activeRuns.set(run.id, {
        handle,
        settle: () => {
          if (run.status === 'running') run.status = 'stopped'
          activeRuns.delete(run.id)
          resolve()
        },
      })
    })
    // 详情回填观测全量字段；runOne 后 current 可能被整个替换，调用方须按 id 重新取 run
    if (current.value?.id === experimentId) {
      try {
        current.value = await api.getExperiment(experimentId)
      } catch {
        // 详情拉取失败不阻塞队列，观测区降级为空态
      }
    }
    // HTTP 层失败（409/404/模型未加载）后端不改 run 状态，详情会带回旧值；
    // 把本地 error 结论写回刷新后的对象，避免「失败」被冲掉（M6 实测坑点）
    if (failed) {
      const fresh = current.value?.runs.find((r) => r.id === run.id)
      if (fresh && fresh.status !== 'error') {
        fresh.status = 'error'
        fresh.error = failed
      }
    }
  }

  // 串行队列：逐个跑 pending 的 run；单个失败不影响后续（各栏独立）
  async function startQueue() {
    if (queueRunning.value || !current.value) return
    queueRunning.value = true
    try {
      const pendingIds = current.value.runs
        .filter((r) => r.status === 'pending')
        .map((r) => r.id)
      for (const id of pendingIds) {
        if (current.value?.id === undefined) break
        const run = current.value.runs.find((r) => r.id === id)
        if (!run || run.status !== 'pending') continue
        await runOne(run)
      }
    } finally {
      queueRunning.value = false
    }
  }

  // 失败/已停止的单栏重试（决策 E）；不复位其它栏
  async function retryRun(runId: number) {
    if (queueRunning.value || !current.value) return
    const run = current.value.runs.find((r) => r.id === runId)
    if (!run || (run.status !== 'error' && run.status !== 'stopped')) return
    queueRunning.value = true
    try {
      run.status = 'pending'
      await runOne(run)
    } finally {
      queueRunning.value = false
    }
  }

  // 停止当前生成中的 run：abort 本地流 + 通知后端中断，settle 让队列继续
  async function stopRun(runId: number) {
    const experiment = current.value
    if (!experiment) return
    const active = activeRuns.get(runId)
    active?.handle.abort()
    try {
      await api.stopRun(experiment.id, runId)
    } catch {
      // 后端可能已结束（竞态），本地状态以 settle 为准
    }
    active?.settle()
  }

  function clearCurrent() {
    current.value = null
  }

  return {
    experiments,
    current,
    queueRunning,
    loadExperiments,
    createExperiment,
    openExperiment,
    removeExperiment,
    startQueue,
    retryRun,
    stopRun,
    clearCurrent,
  }
})

// 让 Pinia store 支持热更新，避免 dev 下新旧代码混跑（生产无影响）
if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useLabStore, import.meta.hot))
}
