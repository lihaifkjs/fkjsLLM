// 模型状态机镜像：unloaded|loading|loaded|unloading|error（技术方案 §3.4）
// 5s 轮询 status + 开关操作后立刻刷新（技术方案 §5.4）
import { acceptHMRUpdate, defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { api, type ModelStatus } from '../api'

const POLL_INTERVAL_MS = 5000

export const useModelStore = defineStore('model', () => {
  const status = ref<ModelStatus>({ state: 'unloaded', vram_used_mb: null, error: null })

  const state = computed(() => status.value.state)
  // 加载/卸载中禁止重复操作（PRD 3.8）
  const busy = computed(() => state.value === 'loading' || state.value === 'unloading')
  const loaded = computed(() => state.value === 'loaded')

  let timer: ReturnType<typeof setInterval> | null = null

  async function refresh() {
    try {
      status.value = await api.getModelStatus()
    } catch {
      // 轮询失败不打断界面，保留上次状态
    }
  }

  function startPolling() {
    if (timer) return
    void refresh()
    timer = setInterval(() => void refresh(), POLL_INTERVAL_MS)
  }

  function stopPolling() {
    if (timer) {
      clearInterval(timer)
      timer = null
    }
  }

  async function load() {
    if (busy.value) return
    status.value = { ...status.value, state: 'loading' }
    try {
      status.value = await api.loadModel()
    } catch (e) {
      status.value = { ...status.value, state: 'error', error: (e as { message?: string }).message ?? '加载失败' }
    }
  }

  async function unload() {
    if (busy.value) return
    status.value = { ...status.value, state: 'unloading' }
    try {
      status.value = await api.unloadModel()
    } catch (e) {
      await refresh()
      throw e
    }
  }

  return { status, state, busy, loaded, refresh, startPolling, stopPolling, load, unload }
})

// 让 Pinia store 支持热更新，避免 dev 下新旧代码混跑（生产无影响）
if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useModelStore, import.meta.hot))
}
