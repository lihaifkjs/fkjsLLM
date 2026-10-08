// 推理参数：默认值对齐技术方案 §3.3（Qwen3 官方推荐），面板见 views/ParamsView.vue
import { acceptHMRUpdate, defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { api, type InferenceParams } from '../api'

// 模型上下文窗口：后端暂无查询接口，常量须与 server/config.py 的 n_ctx 保持一致
export const MODEL_N_CTX = 8192

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

  // 上下文可用预算：总窗口减去本轮最大输出（对齐后端滑窗 budget，server/chat/engine.py）
  const contextBudget = computed(() => MODEL_N_CTX - params.value.max_tokens)

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
