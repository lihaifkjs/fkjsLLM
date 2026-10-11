// 推理参数：默认值对齐技术方案 §3.3（Qwen3 官方推荐），面板见 views/ParamsView.vue
import { acceptHMRUpdate, defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { api, type InferenceParams } from '../api'
import { useModelStore } from './model'

// n_ctx 兜底值：真实值由后端 GET /api/model/status 下发（对齐 server/config.py）
const DEFAULT_N_CTX = 8192

const DEFAULTS: InferenceParams = {
  temperature: 0.7,
  top_p: 0.8,
  top_k: 20,
  repetition_penalty: 1.05,
  max_tokens: 2048,
  system_prompt: '',
  context_strategy: 'sliding_window',
  preset: '',
}

export const useSettingsStore = defineStore('settings', () => {
  const params = ref<InferenceParams>({ ...DEFAULTS })

  // 上下文可用预算：总窗口减去本轮最大输出（对齐后端滑窗 budget，server/chat/engine.py）
  // n_ctx 取模型状态轮询下发的值（5s 周期），未拿到时用兜底值
  const modelStore = useModelStore()
  const contextBudget = computed(
    () => (modelStore.status.n_ctx ?? DEFAULT_N_CTX) - params.value.max_tokens,
  )

  async function load() {
    params.value = await api.getParams()
  }

  // 部分字段补丁；返回服务端展开/匹配后的全量（含 preset）
  async function save(patch: Partial<InferenceParams>) {
    params.value = await api.saveParams(patch)
  }

  return { params, contextBudget, load, save }
})

// 让 Pinia store 支持热更新，避免 dev 下新旧代码混跑（生产无影响）
if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useSettingsStore, import.meta.hot))
}
