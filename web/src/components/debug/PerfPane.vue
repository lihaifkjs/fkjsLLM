<script setup lang="ts">
// 调试抽屉「性能指标」（M5）：TTFT / prefill / decode 分段耗时 + 聚合输出速度
// 数据源：observability 端点的服务端计时真值（llama_perf_context，不含网络延迟，历史消息可查）；
// 生成中或取数失败时降级到帧时间戳统计（含网络延迟，仅供实时参考）
import { computed, toRef } from 'vue'
import type { ChatMessage } from '../../api'
import { useObservability } from '../../composables/useObservability'
import { useDebugStore } from '../../stores/debug'
import { frameStats } from '../../utils/frameStats'

const props = defineProps<{ message: ChatMessage | null }>()

const { observability } = useObservability(toRef(props, 'message'))
const debugStore = useDebugStore()

const stats = computed(() =>
  frameStats(props.message ? debugStore.framesOf(props.message.id) : []),
)

const timing = computed(() => observability.value?.timing ?? null)

// 聚合输出速度：output_tokens / decode 耗时（秒换算），保留一位小数
const tokensPerSec = computed(() => {
  const o = observability.value
  if (!o?.timing || o.output_tokens == null || o.timing.decode_ms <= 0) return null
  return Math.round((o.output_tokens / (o.timing.decode_ms / 1000)) * 10) / 10
})

function fmtMs(ms: number | null): string {
  return ms === null ? '—' : ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`
}
</script>

<template>
  <el-descriptions v-if="timing" :column="1" border>
    <el-descriptions-item label="首 token 延迟（TTFT）">{{ fmtMs(timing.ttft_ms) }}</el-descriptions-item>
    <el-descriptions-item label="prefill 耗时">{{ fmtMs(timing.prefill_ms) }}</el-descriptions-item>
    <el-descriptions-item label="decode 耗时">{{ fmtMs(timing.decode_ms) }}</el-descriptions-item>
    <el-descriptions-item label="输出速度">
      {{ tokensPerSec === null ? '—' : `${tokensPerSec} token/s` }}
    </el-descriptions-item>
  </el-descriptions>

  <el-descriptions v-else-if="stats.ttftMs !== null" :column="1" border>
    <el-descriptions-item label="首 token 延迟（TTFT，含网络）">{{ fmtMs(stats.ttftMs) }}</el-descriptions-item>
    <el-descriptions-item label="全程耗时（含网络）">{{ fmtMs(stats.totalMs) }}</el-descriptions-item>
    <el-descriptions-item label="输出速度（实时）">
      {{ stats.tokensPerSec === null ? '—' : `${stats.tokensPerSec} token/s` }}
    </el-descriptions-item>
  </el-descriptions>
  <el-empty v-else description="该消息无性能数据（帧记录仅在本次会话内可用）" />
</template>
