// 推理参数：默认值对齐技术方案 §3.3（Qwen3 官方推荐）；面板 UI 留 M3
import { acceptHMRUpdate, defineStore } from 'pinia'
import { ref } from 'vue'
import { api, type InferenceParams } from '../api'

const DEFAULTS: InferenceParams = {
  temperature: 0.7,
  top_p: 0.8,
  top_k: 20,
  repetition_penalty: 1.05,
  max_tokens: 2048,
  system_prompt: '',
  preset: '',
}

export const useSettingsStore = defineStore('settings', () => {
  const params = ref<InferenceParams>({ ...DEFAULTS })

  async function load() {
    params.value = await api.getParams()
  }

  async function save(next: InferenceParams) {
    params.value = await api.saveParams(next)
  }

  return { params, load, save }
})

// 让 Pinia store 支持热更新，避免 dev 下新旧代码混跑（生产无影响）
if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useSettingsStore, import.meta.hot))
}
