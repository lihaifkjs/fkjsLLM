// M5 观测数据获取：供调试抽屉各 pane 复用
// 策略：message.id 为真实后端 id（正数，临时占位 id 为负数）时拉取 observability 端点；
// 请求失败（404/网络）或字段为 null（生成中未落库、M5 前老消息）时返回 null，pane 降级到帧统计
import { ref, watch, type Ref } from 'vue'
import { api, type ChatMessage, type Observability } from '../api'
import { useSessionStore } from '../stores/session'

export function useObservability(message: Ref<ChatMessage | null>) {
  const sessionStore = useSessionStore()
  const observability = ref<Observability | null>(null)
  const loading = ref(false)

  watch(
    () => message.value?.id,
    async (id) => {
      observability.value = null
      const sessionId = sessionStore.sessionId
      if (id === undefined || id < 0 || sessionId === null) return
      loading.value = true
      try {
        observability.value = await api.getObservability(sessionId, id)
      } catch {
        // 静默降级：观测数据是增强信息，失败不影响抽屉其它 pane
        observability.value = null
      } finally {
        loading.value = false
      }
    },
    { immediate: true },
  )

  return { observability, loading }
}
